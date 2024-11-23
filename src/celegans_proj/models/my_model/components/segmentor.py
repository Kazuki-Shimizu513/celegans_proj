
import numpy as np
import torch as th
from torch import nn 
from clustering import KMeans as th_KMeans
# from .clustering import KMeans as th_KMeans

import tensorflow as tf
from sklearn.cluster import KMeans
from collections import defaultdict

# TODO:: np to torch
class DiffSeg(nn.Module):
    def __init__(self, kl_threshold, refine, num_points):
        super().__init__()
        # Generate the  gird
        self.grid = self.generate_sampling_grid(num_points)
        # Inialize other parameters 
        # self.kl_threshold = np.array(kl_threshold)
        self.kl_threshold = th.tensor(kl_threshold)
        self.refine = refine

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
        # x_new = np.linspace(start_point, total_len+start_point, num_of_points)
        x_new = th.linspace(start_point, total_len+start_point, steps=num_of_points)
        # y_new = np.linspace(start_point, total_len+start_point, num_of_points)
        y_new = th.linspace(start_point, total_len+start_point, steps=num_of_points)
        # x_new,y_new=np.meshgrid(x_new,y_new,indexing='ij')
        x_new,y_new=th.meshgrid(x_new,y_new,indexing='ij')
        # points = np.concatenate(([x_new.reshape(-1,1),y_new.reshape(-1,1)]),axis=-1).astype(int)
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
        # return np.array(M_list)
        M_list = th.stack(M_list, dim=0)
        return M_list
 
    def get_weight_rato(self, weight_list):
        # This function assigns proportional aggergation weight 
        sizes = []
        for weights in weight_list:
          sizes.append(np.sqrt(weights.shape[-2]))
          # sizes.append(th.sqrt(weights.shape[-2]))
        denom = np.sum(sizes)
        # denom = th.sum(sizes)
        return sizes / denom

    def aggregate_weights(self, weight_list, weight_ratio=None):
        if weight_ratio is None:
          weight_ratio = self.get_weight_rato(weight_list)
        # aggre_weights = np.zeros((64,64,64,64))
        aggre_weights = th.zeros(64,64,64,64)

        for index,weights in enumerate(weight_list):
          # print(f"{weights.shape=}")
          size = int(np.sqrt(weights.shape[-1]))
          # size = int(th.sqrt(weights.shape[-1]))
          ratio = int(64/size)
          # Average over the multi-head channel
          # print(f"in aggregate_weights {ratio=}=int(64/{size})\t{weights.shape=}") # ratio=16=int(64/4) weights.shape=(10, 16, 16)
          weights = weights.mean(0).reshape(-1,size,size)
          # print(f"in aggregate_weights after reshape {weights.shape=}") # (16, 4, 4)
          # Upsample the last two dimensions to 64 x 64
          # print(f"in aggregate_weights before upsampling:{weights.shape=}") # (16, 4, 4)
          # weights = tf.keras.layers.UpSampling2D(
          #     size=(ratio, ratio), data_format="channels_last", interpolation='bilinear'
          # )(tf.expand_dims(weights,axis=-1))
          weights = nn.Upsample(scale_factor=(ratio,ratio), mode='bilinear', align_corners=True
            )(weights.unsqueeze(0))
          # print(f"in aggregate_weights after upsampling:{weights.shape=}") # TensorShape([16, 64, 64, 1])
          # print(f"in aggregate_weights before reshape&normalize:{weights.shape=}") # TensorShape([16, 64, 64, 1])
          # weights = tf.reshape(weights,(size,size,64,64))
          weights = th.reshape(weights,(size,size,64,64))

          # Normalize to make sure each map sums to one
          # weights = weights/tf.math.reduce_sum(weights,(2,3),keepdims=True)
          weights = weights/th.sum(weights,(2,3),keepdims=True)
          # print(f"in aggregate_weights after reshape&normalize:{weights.shape=}") #TensorShape([4, 4, 64, 64])
          
          # print(f"in aggregate_weights before spatial tiling:{weights.shape=}") # TensorShape([4, 4, 64, 64])
          # Spatial tiling along the first two dimensions
          # weights = tf.repeat(weights,repeats=ratio,axis=0)
          # weights = tf.repeat(weights,repeats=ratio,axis=1)
          weights = weights.repeat(ratio,ratio,1,1)
          # print(f"in aggregate_weights after spatial tiling:{weights.shape=}") # TensorShape([64, 64, 64, 64])
          # print()

          # Aggrgate accroding to weight_ratio
          aggre_weights += weights*weight_ratio[index]
        # return aggre_weights.numpy().astype(np.double)
        return aggre_weights

    def KL(self,x,Y):
        # qoutient = tf.math.log(x)-tf.math.log(Y)
        # kl_1 = tf.math.reduce_sum(tf.math.multiply(x, qoutient),axis=(-2,-1))/2
        # kl_2 = -tf.math.reduce_sum(tf.math.multiply(Y, qoutient),axis=(-2,-1))/2
        # return tf.math.add(kl_1,kl_2)
        qoutient = th.log(x)-th.log(Y)
        kl_1 = th.sum(th.mul(x, qoutient),(-2,-1))/2
        kl_2 = -th.sum(th.mul(Y, qoutient),(-2,-1))/2
        return th.add(kl_1,kl_2)


    def mask_merge(self, iter, attns, kl_threshold, grid=None):
        if iter == 0:
          # The first iteration of merging
          anchors = attns[grid[:,0],grid[:,1],:,:] # 256 x 64 x 64
          # anchors = tf.expand_dims(anchors, axis=(1)) # 256 x 1 x 64 x 64
          anchors = th.unsqueeze(anchors, 1) # 256 x 1 x 64 x 64
          attns = attns.reshape(1,4096,64,64) 
          # 256 x 4096 x 64 x 64 is too large for a single gpu, splitting into 16 portions
          split = np.sqrt(grid.shape[0]).astype(int)
          kl_bin=[]
          for i in range(split):
            # temp = self.KL(tf.cast(anchors[i*split:(i+1)*split],tf.float16),
            #                tf.cast(attns,tf.float16)
            #         ) < kl_threshold[iter] # type cast from tf.float64 to tf.float16
            temp = self.KL(anchors[i*split:(i+1)*split].type(th.float16),
                           attns.type(th.float16)
                    ) < kl_threshold[iter] # type cast from tf.float64 to tf.float16
            # print(f"In mask merge {temp.shape=}")# TensorShape([2, 4096])
            kl_bin.append(temp)
          # kl_bin = tf.cast(tf.concat(kl_bin, axis=0), tf.float64) # 256 x 4096
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
            # kl_bin = (self.KL(anchor,attns) < kl_threshold[iter]).numpy() # 64 x 64
            kl_bin = self.KL(anchor,attns) < kl_threshold[iter]# 64 x 64
            if kl_bin.sum() > 0:
              # matched_idx = np.arange(len(attns))[kl_bin.reshape(-1)]
              matched_idx = th.arange(len(attns))[kl_bin.reshape(-1)]
              for idx in matched_idx: matched.add(idx)
              aggregated_attn = attns[kl_bin].mean(0)
              new_attns.append(aggregated_attn.reshape(1,64,64))
          new_attns = th.stack(new_attns, dim=0)

        # print(f"In mask merge {new_attns.shape=}")# TensorShape([4, 64,64])
        # return np.array(new_attns)
        return new_attns

    def generate_masks(self, attns, kl_threshold, grid):
        # Iterative Attention Merging
        for i in range(len(kl_threshold)):
          if i == 0:
            attns_merged = self.mask_merge(i, attns, kl_threshold, grid=grid)
          else:
            attns_merged = self.mask_merge(i, attns_merged, kl_threshold)

        # print(attns_merged.shape, attns_merged.ndim, type(attns_merged)) # (144, 64, 64), 3, np.ndarry
        # attns_merged = attns_merged[:,0,:,:]

        # Kmeans refinement (optional for better visual consistency)
        if self.refine:
          attns = attns.reshape(-1,64*64)
          #  X = None, k=2, n_iters = 10, p = 2
          # kmeans = KMeans(n_clusters=attns_merged.shape[0], init=attns_merged.reshape(-1,64*64), n_init=1).fit(attns)
          # clusters = kmeans.labels_
          kmeans = th_KMeans(k=attns_merged.shape[0], n_iters = 100, p = 2,)
          kmeans.train(attns)
          clusters = kmeans.labels
          num_clusters = kmeans.k
          # print(f"In th_Kmeans {attns.shape=} {attns_merged.shape=}") # 4096
          # print(f"In th_Kmeans {clusters.shape=} {num_clusters=}=={len(set(clusters))=}") # 4096
          # print(f"In th_Kmeans {clusters.min()=}{clusters.max()=}") # 4096
          # print()
          attns_merged = []
          for i in range(num_clusters):
            cluster = (i == clusters)
            attns_merged.append(attns[cluster,:].mean(0).reshape(64,64))
          # attns_merged = np.array(attns_merged)
          attns_merged = th.stack(attns_merged, dim=0)
          # print(f"In th_Kmeans {attns_merged.shape=}") # 4096,64,64

        # print(attns_merged.shape, attns_merged.ndim, type(attns_merged)) 
        # Upsampling
        # self.upsampled = tf.keras.layers.UpSampling2D(size=(8, 8), data_format="channels_last", interpolation='bilinear')(tf.expand_dims(attns_merged,axis=-1))
        # self.upsampled = tf.keras.layers.UpSampling2D(size=(4, 4), data_format="channels_last", interpolation='bilinear')(tf.expand_dims(attns_merged,axis=-1))
        self.upsampled = nn.Upsample(scale_factor=(4,4), mode='bilinear', align_corners=True
          )(attns_merged.unsqueeze(0))
 
        # Non-Maximum Suppression
        # M_final = tf.reshape(tf.math.argmax(self.upsampled,axis=0),(512,512)).numpy()
        # M_final = tf.reshape(tf.math.argmax(self.upsampled,axis=0),(256,256)).numpy()
        M_final = th.reshape(th.argmax(self.upsampled,dim=1),(256,256))
 
        # print(f"{M_final.shape=}")

        return M_final

class DiffSeg_TF:
  def __init__(self, kl_threshold, refine, num_points):
    # Generate the  gird
    self.grid = self.generate_sampling_grid(num_points)
    # Inialize other parameters 
    self.kl_threshold = np.array(kl_threshold)
    self.refine = refine

  def generate_sampling_grid(self,num_of_points):
    segment_len = 63//(num_of_points-1)
    total_len = segment_len*(num_of_points-1)
    start_point = (63 - total_len)//2
    x_new = np.linspace(start_point, total_len+start_point, num_of_points)
    y_new = np.linspace(start_point, total_len+start_point, num_of_points)
    x_new,y_new=np.meshgrid(x_new,y_new,indexing='ij')
    points = np.concatenate(([x_new.reshape(-1,1),y_new.reshape(-1,1)]),axis=-1).astype(int)
    return points
  
  def get_weight_rato(self, weight_list):
    # This function assigns proportional aggergation weight 
    sizes = []
    for weights in weight_list:
      sizes.append(np.sqrt(weights.shape[-2]))
    denom = np.sum(sizes)
    return sizes / denom

  def aggregate_weights(self, weight_list, weight_ratio=None):
    if weight_ratio is None:
      weight_ratio = self.get_weight_rato(weight_list)
    aggre_weights = np.zeros((64,64,64,64))

    for index,weights in enumerate(weight_list):
      size = int(np.sqrt(weights.shape[-1]))
      ratio = int(64/size)
      # Average over the multi-head channel
      # print(f"in aggregate_weights {ratio=}=int(64/{size})\t{weights.shape=}") # ratio=16=int(64/4) weights.shape=(10, 16, 16)
      weights = weights.mean(0).reshape(-1,size,size)
      # print(f"in aggregate_weights after reshape {weights.shape=}") # (16, 4, 4)
      # Upsample the last two dimensions to 64 x 64
      # print(f"in aggregate_weights before upsampling:{weights.shape=}") # (16, 4, 4)
      weights = tf.keras.layers.UpSampling2D(size=(ratio, ratio), data_format="channels_last", interpolation='bilinear')(tf.expand_dims(weights,axis=-1))
      # print(f"in aggregate_weights after upsampling:{weights.shape=}") # TensorShape([16, 64, 64, 1])
      # print(f"in aggregate_weights before reshape&normalize:{weights.shape=}") # TensorShape([16, 64, 64, 1])
      weights = tf.reshape(weights,(size,size,64,64))

      # Normalize to make sure each map sums to one
      weights = weights/tf.math.reduce_sum(weights,(2,3),keepdims=True)
      # print(f"in aggregate_weights after reshape&normalize:{weights.shape=}") #TensorShape([4, 4, 64, 64])
      
      # print(f"in aggregate_weights before spatial tiling:{weights.shape=}") # TensorShape([4, 4, 64, 64])
      # Spatial tiling along the first two dimensions
      weights = tf.repeat(weights,repeats=ratio,axis=0)
      weights = tf.repeat(weights,repeats=ratio,axis=1)
      # print(f"in aggregate_weights after spatial tiling:{weights.shape=}") # TensorShape([64, 64, 64, 64])
      # print()

      # Aggrgate accroding to weight_ratio
      aggre_weights += weights*weight_ratio[index]
    return aggre_weights.numpy().astype(np.double)

  def aggregate_x_weights(self, weight_list, weight_ratio=None):
    # x_weights: 8 x size**2 x 77
    # return 512 x 512 x 77
    if weight_ratio is None:
      weight_ratio = self.get_weight_rato(weight_list)
    aggre_weights = np.zeros((512, 512, 77))

    for index,weights in enumerate(weight_list):
      size = int(np.sqrt(weights.shape[-2]))
      ratio = int(512/size)
      weights = weights.mean(0).reshape(1,size,size,-1)
      weights = tf.keras.layers.UpSampling2D(size=(ratio, ratio), data_format="channels_last", interpolation='bilinear')(weights)
      weights = weights/tf.math.reduce_sum(weights,axis=-1,keepdims=True)
      aggre_weights += weights*weight_ratio[index]
    return aggre_weights.numpy().astype(np.double)

  def KL(self,x,Y):
      qoutient = tf.math.log(x)-tf.math.log(Y)
      kl_1 = tf.math.reduce_sum(tf.math.multiply(x, qoutient),axis=(-2,-1))/2
      kl_2 = -tf.math.reduce_sum(tf.math.multiply(Y, qoutient),axis=(-2,-1))/2
      return tf.math.add(kl_1,kl_2)


  def mask_merge(self, iter, attns, kl_threshold, grid=None):
    if iter == 0:
      # The first iteration of merging
      anchors = attns[grid[:,0],grid[:,1],:,:] # 256 x 64 x 64
      anchors = tf.expand_dims(anchors, axis=(1)) # 256 x 1 x 64 x 64
      attns = attns.reshape(1,4096,64,64) 
      # 256 x 4096 x 64 x 64 is too large for a single gpu, splitting into 16 portions
      split = np.sqrt(grid.shape[0]).astype(int)
      kl_bin=[]
      for i in range(split):
        temp = self.KL(tf.cast(anchors[i*split:(i+1)*split],tf.float16),
                       tf.cast(attns,tf.float16)
                ) < kl_threshold[iter] # type cast from tf.float64 to tf.float16

        kl_bin.append(temp)
        # print(f"In mask merge: {temp.shape=}")
      kl_bin = tf.cast(tf.concat(kl_bin, axis=0), tf.float64) # 256 x 4096
      # print(f"In mask merge {kl_bin.shape=}")
      # print(f"In mask merge {attns.shape=}")
      # print(f"In mask merge {tf.reshape(attns,(-1,4096)).shape=}")
      new_attns = tf.reshape(tf.matmul(kl_bin,tf.reshape(attns,(-1,4096)))/tf.math.reduce_sum(kl_bin,1,keepdims=True),(-1,64,64)) # 256 x 64 x 64

    else:
      # The rest of merging iterations, reducing the number of masks
      matched = set()
      new_attns = []
      for i,point in enumerate(attns):
        if i in matched:
          continue
        matched.add(i)
        anchor = point
        kl_bin = (self.KL(anchor,attns) < kl_threshold[iter]).numpy() # 64 x 64
        if kl_bin.sum() > 0:
          matched_idx = np.arange(len(attns))[kl_bin.reshape(-1)]
          for idx in matched_idx: matched.add(idx)
          aggregated_attn = attns[kl_bin].mean(0)
          new_attns.append(aggregated_attn.reshape(1,64,64))
      new_attns = np.array(new_attns)
    # print(f"In mask merge {new_attns.shape=}")# TensorShape([4, 64,64])
    return np.array(new_attns)

  def generate_masks(self, attns, kl_threshold, grid):
    # Iterative Attention Merging
    for i in range(len(kl_threshold)):
      if i == 0:
        attns_merged = self.mask_merge(i, attns, kl_threshold, grid=grid)
      else:
        attns_merged = self.mask_merge(i, attns_merged, kl_threshold)

    # print(attns_merged.shape, attns_merged.ndim, type(attns_merged)) # (144, 64, 64), 3, np.ndarry
    # attns_merged = attns_merged[:,0,:,:]

    # Kmeans refinement (optional for better visual consistency)
    if self.refine:
      attns = attns.reshape(-1,64*64)
      kmeans = KMeans(n_clusters=attns_merged.shape[0], init=attns_merged.reshape(-1,64*64), n_init=1).fit(attns)
      clusters = kmeans.labels_
      # print(f"In Kmeans {clusters.shape=}")# =(4096,)
      # print(f"In Kmeans {attns.shape=} {attns_merged.shape=}") # 4096
      # print(f"In Kmeans {clusters.shape=}") # 4096
      attns_merged = []
      for i in range(len(set(clusters))):
        cluster = (i == clusters)
        attns_merged.append(attns[cluster,:].mean(0).reshape(64,64))
      attns_merged = np.array(attns_merged)

    # print(attns_merged.shape, attns_merged.ndim, type(attns_merged)) # (144, 64, 64), 3, np.ndarry
    # Upsampling
    # self.upsampled = tf.keras.layers.UpSampling2D(size=(8, 8), data_format="channels_last", interpolation='bilinear')(tf.expand_dims(attns_merged,axis=-1))
    self.upsampled = tf.keras.layers.UpSampling2D(size=(4, 4), data_format="channels_last", interpolation='bilinear')(tf.expand_dims(attns_merged,axis=-1))

    # print(f"{self.upsampled.shape=}{self.upsampled.ndim, type(self.upsampled)}")
    # Non-Maximum Suppression
    # M_final = tf.reshape(tf.math.argmax(self.upsampled,axis=0),(512,512)).numpy()
    M_final = tf.reshape(tf.math.argmax(self.upsampled,axis=0),(256,256)).numpy()
    # print(f"{M_final.shape=}")

    return M_final
  
  def segment(self, weight_64, weight_32, weight_16, weight_8, weight_ratio = None):
    M_list = []
    for i in range(len(weight_64)):
      # Step 1: Attention Aggregation
      weights = self.aggregate_weights([weight_64[i],weight_32[i], weight_16[i], weight_8[i]],weight_ratio=weight_ratio)
      # Step 2 & 3: Iterative Merging & NMS
      M_final = self.generate_masks(weights, self.kl_threshold, self.grid)
      M_list.append(M_final)
    return np.array(M_list)

  def get_semantics(self, pred, x_weight, nouns, voting="majority"):
        # This function assigns semantic labels to masks 
        indices = [item[0]+1 for item in nouns] # Igonore the first BOS token
        prompt_list = [item[1] for item in nouns]
        x_weight = x_weight[:,:,indices] # size x size x N
        x_weight = x_weight.reshape(512*512,-1)
        norm = np.linalg.norm(x_weight,axis=0,keepdims=True)
        x_weight = x_weight/norm # Normalize the cross-attention maps spatially
        pred = pred.reshape(512*512,-1)

        label_to_mask = defaultdict(list)
        for i in set(pred.flatten()):
          if voting == "majority":
            logits = x_weight[(pred==i).flatten(),:]
            index = logits.argmax(axis=-1)
            category = prompt_list[int(np.median(index))]
          else:
            logit = x_weight[(pred==i).flatten(),:].mean(0)
            category = prompt_list[logit.argmax(axis=-1)]
          label_to_mask[category].append(i)
        return label_to_mask


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

    segmentor = DiffSeg_TF(
        kl_threshold=[0.002]*4,
        refine=True,
        num_points=2,
    )

    pred_mask = segmentor.segment(
        weights_dict["weight_32"].numpy(force=False), 
        weights_dict["weight_16"].numpy(force=False), 
        weights_dict["weight_8"].numpy(force=False), 
        weights_dict["weight_4"].numpy(force=False), 
        weight_ratio = None
    )


    print(f"{pred_mask.shape=}\t{type(pred_mask)=}{pred_mask.min()=}{pred_mask.max()=}")

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

