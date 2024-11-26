
import numpy as np
import torch as th
from torch import nn 

# from clustering import KMeans as th_KMeans
from .clustering import KMeans as th_KMeans

class DiffSeg(nn.Module):
    def __init__(self, 
        kl_threshold, 
        num_points, 
        refine=True,
        n_iters= 100, 
        p= 2,
        device = "cuda",
):
        super().__init__()
        # Generate the  gird
        self.grid = self.generate_sampling_grid(num_points)
        # Inialize other parameters 
        self.kl_threshold = kl_threshold
        self.refine = refine
        self.n_iters= n_iters
        self.p=p 
        self.device = device

    def set_KL_THRESHOLD(self,new_KL_THRESHOLD):
        self.kl_threshold = new_KL_THRESHOLD

    def forward(self, weight_64, weight_32, weight_16, weight_8, weight_ratio = None ):
        preds = self.segment(
            weight_64,
            weight_32, 
            weight_16,
            weight_8,
            weight_ratio = None
        )
        return preds

    def generate_sampling_grid(self,num_of_points):
        segment_len = 63//(num_of_points-1)
        total_len = segment_len*(num_of_points-1)
        start_point = (63 - total_len)//2
        x_new = th.linspace(start_point, total_len+start_point, steps=num_of_points)
        y_new = th.linspace(start_point, total_len+start_point, steps=num_of_points)
        x_new,y_new=th.meshgrid(x_new,y_new,indexing='ij')
        points = th.cat(([x_new.reshape(-1,1),y_new.reshape(-1,1)]),-1).type(th.int32)
        return points
     
    def segment(self, weight_64, weight_32, weight_16, weight_8, weight_ratio = None):
        M_list = []
        for i in range(len(weight_64)):
          # Step 1: Attention Aggregation
          weights = self.aggregate_weights([weight_64[i],weight_32[i], weight_16[i], weight_8[i]],weight_ratio=weight_ratio)
          # Step 2 & 3: Iterative Merging & NMS
          M_final = self.generate_masks(weights, self.kl_threshold, self.grid)
          M_list.append(M_final)
        M_list = th.stack(M_list, dim=0)
        return M_list
 
    @staticmethod
    def get_weight_rato(weight_list):
        # This function assigns proportional aggergation weight 
        sizes = []
        for weights in weight_list:
          sizes.append(np.sqrt(weights.shape[-2]))
        denom = np.sum(sizes)
        return sizes / denom

    def aggregate_weights(self, weight_list, weight_ratio=None):
        if weight_ratio is None:
          weight_ratio = self.get_weight_rato(weight_list)
        aggre_weights = th.zeros(64,64,64,64, device=self.device)

        for index,weights in enumerate(weight_list):
          size = int(np.sqrt(weights.shape[-1]))
          ratio = int(64/size)
          # Average over the multi-head channel
          weights = weights.mean(0).reshape(-1,size,size)
          # Upsample the last two dimensions to 64 x 64
          weights = nn.Upsample(scale_factor=(ratio,ratio), mode='bilinear', align_corners=True)(weights.unsqueeze(0))
          weights = th.reshape(weights,(size,size,64,64))

          # Normalize to make sure each map sums to one
          weights = weights/th.sum(weights,(2,3),keepdims=True)
          
          # Spatial tiling along the first two dimensions
          weights = weights.repeat(ratio,ratio,1,1)


          # Aggrgate accroding to weight_ratio
          aggre_weights += weights*weight_ratio[index]
        return aggre_weights

    def KL(self,x,Y):
        qoutient = th.log(x)-th.log(Y)
        kl_1 = th.sum(th.mul(x, qoutient),(-2,-1))/2
        kl_2 = -th.sum(th.mul(Y, qoutient),(-2,-1))/2
        return th.add(kl_1,kl_2)


    def mask_merge(self, iter, attns, kl_threshold, grid=None):
        if iter == 0:
          # The first iteration of merging
          anchors = attns[grid[:,0],grid[:,1],:,:] # 256 x 64 x 64
          anchors = th.unsqueeze(anchors, 1) # 256 x 1 x 64 x 64
          attns = attns.reshape(1,4096,64,64) 
          # 256 x 4096 x 64 x 64 is too large for a single gpu, splitting into 16 portions
          split = np.sqrt(grid.shape[0]).astype(int)
          kl_bin=[]
          for i in range(split):
            temp = self.KL(anchors[i*split:(i+1)*split].type(th.float16),
                           attns.type(th.float16)
                    ) < kl_threshold[iter] # type cast from tf.float64 to tf.float16
            kl_bin.append(temp)
          kl_bin = th.cat(kl_bin, dim=0)# ,type(th.float64) # 256 x 4096
          kl_num = th.where(kl_bin, 1., 0.)
          new_attns = th.reshape(
              th.matmul(kl_num,attns.reshape(-1,4096))/th.sum(kl_bin,1,keepdims=True),
              (-1,64,64)
          )# 256 x 64 x 64
        else:
          # The rest of merging iterations, reducing the number of masks
          matched = set()
          new_attns = []
          for i,point in enumerate(attns):
            if i in matched:
              continue
            matched.add(i)
            anchor = point
            kl_bin = self.KL(anchor,attns) < kl_threshold[iter]# 64 x 64
            if kl_bin.sum() > 0:
              matched_idx = th.arange(len(attns), device=self.device)[kl_bin.reshape(-1)]
              for idx in matched_idx: matched.add(idx)
              aggregated_attn = attns[kl_bin].mean(0)
              new_attns.append(aggregated_attn.reshape(1,64,64))
          new_attns = th.stack(new_attns, dim=0)

        return new_attns

    def generate_masks(self, attns, kl_threshold, grid):
        # Iterative Attention Merging
        for i in range(len(kl_threshold)):
          if i == 0:
            attns_merged = self.mask_merge(i, attns, kl_threshold, grid=grid)
          else:
            attns_merged = self.mask_merge(i, attns_merged, kl_threshold)

        # attns_merged = attns_merged[:,0,:,:]

        # Kmeans refinement (optional for better visual consistency)
        if self.refine:
          attns = attns.reshape(-1,64*64)
          kmeans = th_KMeans(
              k=attns_merged.shape[0], 
              n_iters = self.n_iters,# 100, 
              p = self.p,# 2,
          )
          kmeans.to(device=self.device)
          kmeans.train(attns)
          clusters = kmeans.labels
          num_clusters = kmeans.k
          attns_merged = []
          for i in range(num_clusters):
            cluster = (i == clusters)
            attns_merged.append(attns[cluster,:].mean(0).reshape(64,64))
          attns_merged = th.stack(attns_merged, dim=0)

        # Upsampling
        self.upsampled = nn.Upsample(scale_factor=(4,4), mode='bilinear', align_corners=True
          )(attns_merged.unsqueeze(0))
 
        # Non-Maximum Suppression
        M_final = th.reshape(th.argmax(self.upsampled,dim=1),(256,256))
        return M_final

if __name__ == "__main__":
    B = 2
    C =  20 

    weight_64 = np.random.randn(B,C,64*64,64*64)
    weight_32 = np.random.randn(B,C,32*32,32*32)
    weight_16 = np.random.randn(B,C,16*16,16*16)
    weight_8 = np.random.randn(B,C,8*8,8*8)

    dir_path = "/mnt/c/Users/compbio/Desktop/shimizudata/test"
    pt_path = f"{dir_path}/attention_maps.pt"
    weights_dict = th.load(pt_path, weights_only=True)
    print(weights_dict.keys())

    segmentor = DiffSeg(
        kl_threshold=[0.002]*4,
        refine=True,
        num_points=2,
    )

    pred_mask = segmentor.segment(
        weights_dict["weight_32"], 
        weights_dict["weight_16"], 
        weights_dict["weight_8"], 
        weights_dict["weight_4"], 
        weight_ratio = None
    )

    print(f"{pred_mask.shape=}\t{type(pred_mask)=}{pred_mask.min()=}{pred_mask.max()=}")
    pred_mask = pred_mask.numpy(force=True)

    from PIL import Image
    mask_color = np.linspace(0, 255, pred_mask.max(), dtype=int)
    for idx in range(pred_mask.max()):
        pred_mask = np.where(pred_mask==idx, mask_color[idx], pred_mask)
    # reshape to 2d
    pred_mask = Image.fromarray(np.uint8(np.reshape(pred_mask,(256,256))) , 'L')
    mask_path = f"{dir_path}/pred_mask.png"
    pred_mask.save(mask_path)

