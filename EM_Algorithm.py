import tensorflow as tf;
import math;


def EM_Algoritm(sample, mixture_Count=10, iteration= 100):
    '''
    Searcing the Gaussian-mixture by Expectation-Maximization algorithm
    Args:
        sample: 1d tensor (N).
        mixture_Count: positive integer scalar(M).
        iteration: positive integer scalar.
    Returns:        
        alpha: 1D tensor representing the weights of Gaussian. Size is (M)
        mu: 1D tensor representing the loc of Gaussian. Size is (M)
        sigma: 1D tensor representing the std of Gaussian. Size is (M)
    Refer:
        https://stackoverflow.com/questions/22750634/expectation-maximizationgmm-em-never-finds-the-correct-parameters-mixture-of
        https://www.tensorflow.org/api_docs/python/tf/distributions/Normal        
    '''
    with tf.variable_scope('em_Algorithm'):        
        alpha_Tensor = tf.ones([mixture_Count], dtype=tf.float32);
        mu_Tensor = tf.random_uniform([mixture_Count], dtype=tf.float32);
        sigma_Tensor = tf.ones([mixture_Count], dtype=tf.float32) * 10;   #When sigma is too small, the values become nan.

        def condition(index, alpha_Tensor, mu_Tensor, sigma_Tensor):
            return tf.less(index, iteration);

        def while_Body(index, alpha_Tensor, mu_Tensor, sigma_Tensor):
            tiled_Sample = tf.tile(tf.expand_dims(sample, axis=0), multiples=(mixture_Count, 1))
            tiled_Mu_Tensor = tf.tile(tf.expand_dims(mu_Tensor, axis=1), multiples=(1, tf.shape(sample)[0]))
            tiled_Sigma_Tensor = tf.tile(tf.expand_dims(sigma_Tensor, axis=1), multiples=(1, tf.shape(sample)[0]))    
    
            w_MT = tf.exp(-0.5 * tf.pow(tiled_Sample - tiled_Mu_Tensor, 2) / tf.pow(tiled_Sigma_Tensor, 2)) / tf.sqrt(2 * math.pi * tf.pow(tiled_Sigma_Tensor, 2))
            w_MT /= tf.reduce_sum(w_MT, axis=0);
    
            alpha_Tensor = tf.reduce_sum(w_MT, axis=1) / tf.cast(tf.shape(sample)[0], dtype=tf.float32);
            mu_Tensor = tf.reduce_sum(w_MT * sample, axis=1) / tf.reduce_sum(w_MT, axis=1);
            sigma_Tensor = tf.clip_by_value(tf.sqrt(tf.reduce_sum(w_MT * tf.pow(sample - tf.expand_dims(mu_Tensor, axis=1), 2), axis=1) / tf.reduce_sum(w_MT, axis=1)), clip_value_min=0.1, clip_value_max=np.inf)

            return index+1, alpha_Tensor, mu_Tensor, sigma_Tensor
        
        index = tf.constant(0);
        _, alpha_Tensor, mu_Tensor, sigma_Tensor = tf.while_loop(condition, while_Body, [index, alpha_Tensor, mu_Tensor, sigma_Tensor])
            
        return alpha_Tensor, mu_Tensor, sigma_Tensor;
    
    
def EM_Algoritm_Discret_Data(sample, mixture_Count=10, iteration= 100):
    '''
    Searcing the Gaussian-mixture by Expectation-Maximization algorithm
    Args:
        sample: 1d tensor (L). The 'L' mean 'x' axis. Each cell is how much sample is in the point. Total sample count 'N' is 'sum(range(sample) * sample)'.
        mixture_Count: positive integer scalar(M).
        iteration: positive integer scalar.
    Returns:        
        alpha: 2D tensor representing the weights of Gaussian. Size is (M)
        mu: 2D tensor representing the loc of Gaussian. Size is (M)
        sigma: 2D tensor representing the std of Gaussian. Size is (M)
    Refer:
        https://stackoverflow.com/questions/22750634/expectation-maximizationgmm-em-never-finds-the-correct-parameters-mixture-of
        https://www.tensorflow.org/api_docs/python/tf/distributions/Normal        
    '''
    with tf.variable_scope('em_Algorithm'): 
        alpha_Tensor = tf.ones([mixture_Count], dtype=tf.float32);
        mu_Tensor = tf.random_uniform([mixture_Count], dtype=tf.float32);
        sigma_Tensor = tf.ones([mixture_Count], dtype=tf.float32) * tf.reduce_sum(sample) / 100;   #When sigma is too small, the values become nan.

        x_Range = tf.range(tf.cast(tf.shape(sample)[0], dtype=tf.float32))

        def condition(index, alpha_Tensor, mu_Tensor, sigma_Tensor):
            return tf.less(index, iteration);

        def while_Body(index, alpha_Tensor, mu_Tensor, sigma_Tensor):
            tiled_X_Range = tf.tile(tf.expand_dims(x_Range, axis=0), multiples=(mixture_Count, 1))    #[M, L]   the value is index.
            tiled_Mu_Tensor = tf.tile(tf.expand_dims(mu_Tensor, axis=1), multiples=(1, tf.shape(sample)[0]))    #[M, L]
            tiled_Sigma_Tensor = tf.tile(tf.expand_dims(sigma_Tensor, axis=1), multiples=(1, tf.shape(sample)[0]))    #[M, L]
    
            w_MT = tf.exp(-0.5 * tf.pow(tiled_X_Range - tiled_Mu_Tensor, 2) / tf.pow(tiled_Sigma_Tensor, 2)) / tf.sqrt(2 * math.pi * tf.pow(tiled_Sigma_Tensor, 2))  #[M, L]
            w_MT /= tf.reduce_sum(w_MT, axis=0);
    
            alpha_Tensor = tf.reduce_sum(sample * w_MT, axis=1) / tf.reduce_sum(sample, axis=0);
            mu_Tensor = tf.reduce_sum(x_Range * sample * w_MT, axis=1) / tf.reduce_sum(sample * w_MT, axis=1);
            sigma_Tensor = tf.clip_by_value(
                tf.sqrt(tf.reduce_sum(sample * w_MT * tf.pow(x_Range - tf.expand_dims(mu_Tensor, axis=1), 2), axis=1) / tf.reduce_sum(sample * w_MT, axis=1)),
                clip_value_min=0.1,
                clip_value_max=np.inf
                );

            return index+1, alpha_Tensor, mu_Tensor, sigma_Tensor
        
        index = tf.constant(0);
        _, alpha_Tensor, mu_Tensor, sigma_Tensor = tf.while_loop(condition, while_Body, [index, alpha_Tensor, mu_Tensor, sigma_Tensor])
            
        return alpha_Tensor, mu_Tensor, sigma_Tensor;
