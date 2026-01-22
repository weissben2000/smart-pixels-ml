import tensorflow as tf
import tensorflow_probability as tfp

# custom loss function
def custom_loss(y, p_base, minval=1e-9, maxval=1e9, scale = 512):
    
    p = p_base
    
    mu = p[:, 0:8:2]
    
    # creating each matrix element in 4x4
    Mdia = minval + tf.math.maximum(p[:, 1:8:2], 0.0)
    Mcov = p[:,8:]
    
    # placeholder zero element
    zeros = tf.zeros_like(Mdia[:,0])
    
    # assembles scale_tril matrix
    row1 = tf.stack([Mdia[:,0],zeros,zeros,zeros])
    row2 = tf.stack([Mcov[:,0],Mdia[:,1],zeros,zeros])
    row3 = tf.stack([Mcov[:,1],Mcov[:,2],Mdia[:,2],zeros])
    row4 = tf.stack([Mcov[:,3],Mcov[:,4],Mcov[:,5],Mdia[:,3]])

    scale_tril = tf.transpose(tf.stack([row1,row2,row3,row4]),perm=[2,0,1])

    dist = tfp.distributions.MultivariateNormalTriL(loc = mu, scale_tril = scale_tril) 
    
    likelihood = dist.prob(y)  
    likelihood = tf.clip_by_value(likelihood,minval,maxval)

    NLL = -1*tf.math.log(likelihood)

    return tf.keras.backend.sum(NLL) 

def SSIMLoss(y_true, y_pred, max_val = 3.0):
    # max_val = max(max(y_true), max(y_pred))
    # Calculate SSIM for each image in the batch
    ssim_value = tf.image.ssim(y_true, max_val*y_pred, max_val=max_val)
    # The loss is 1 - mean SSIM across the batch
    return 1.0 - tf.reduce_mean(ssim_value)

def PSNRLoss(y_true, y_pred, max_val=3.0):
    psnr = tf.image.psnr(y_true, max_val*y_pred, max_val=max_val)
    return -tf.reduce_mean(psnr)