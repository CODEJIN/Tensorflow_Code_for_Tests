import tensorflow as tf;

def Correlation2D(x, y):
    """
    Compute the correlations between each rows of two tensors. Main purpose is checking the
        correlations between the units of two layers
    Args:
        x: 2d tensor (MxN). The number of second dimension should be same to y's second dimension.
        y: 2d tensor (LxN). The number of second dimension should be same to x's second dimension.
    Returns:        
        correlation_Tensor: A `Tensor` representing the correlation between the rows. Size is (M x L)
        p_Value_Tensor: A `Tensor` representing the p-value of correlation. Size is (M x L)
    """    
    avgsub_Input_Tensor = x - tf.reduce_mean(x, axis = 1, keepdims=True);  #[M, N]
    avgsub_Hidden_Tensor = y - tf.reduce_mean(y, axis = 1, keepdims=True);  #[L, M]

    sumed_Pow_Input_Tensor = tf.reduce_sum(tf.pow(avgsub_Input_Tensor, 2), axis=1, keepdims= True)      #[M, 1]
    sumed_Pow_Hidden_Tensor = tf.reduce_sum(tf.pow(avgsub_Hidden_Tensor, 2), axis=1, keepdims= True)    #[L, 1]

    correlation_Tensor = tf.matmul(avgsub_Input_Tensor, tf.transpose(avgsub_Hidden_Tensor)) / tf.sqrt(tf.matmul(sumed_Pow_Input_Tensor, tf.transpose(sumed_Pow_Hidden_Tensor)));    #[M, L]
    p_Value_Tensor = 1 - tf.erf(tf.abs(correlation_Tensor) * tf.sqrt(tf.cast(tf.shape(x)[1], tf.float32)) / tf.sqrt(2.0));  #[M, L]

    correlation_Tensor = tf.identity(correlation_Tensor, name="correlation");
    p_Value_Tensor = tf.identity(p_Value_Tensor, name="p_value");

    return (correlation_Tensor, p_Value_Tensor)

if __name__ == "__main__":
    with tf.Session() as tf_Session:
        input_Tensor = tf.placeholder(tf.float32, shape=[None, None, 256]);     #[20, 110, 256]
        hidden_Tensor = tf.placeholder(tf.float32, shape=[None, None, 300]);    #[20, 110, 300]
        reshaped_Input_Tensor = tf.transpose(tf.reshape(input_Tensor, shape=[-1, 256]));    #[256, 2200]
        reshaped_Hidden_Tensor = tf.transpose(tf.reshape(hidden_Tensor, shape=[-1, 300]));   #[300, 2200]
        
        import numpy as np;
        x = np.random.rand(20, 110, 256).astype("float32");
        y = np.random.rand(20, 110, 300).astype("float32");
        correlation_Array, p_Value_Array = tf_Session.run(Correlation2D(reshaped_Input_Tensor, reshaped_Hidden_Tensor), feed_dict = {input_Tensor: x, hidden_Tensor: y});
        
        print(correlation_Array[0,0], p_Value_Array[0,0])
        print(correlation_Array[50, 52], p_Value_Array[50, 52])

        from scipy.stats.stats import pearsonr;
        print(pearsonr(x.reshape([-1, 256]).transpose()[0], y.reshape([-1, 300]).transpose()[0]))
        print(pearsonr(x.reshape([-1, 256]).transpose()[50], y.reshape([-1, 300]).transpose()[52]))
