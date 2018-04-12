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
    avgsub_X_Tensor = x - tf.reduce_mean(x, axis = 1, keepdims=True);  #[M, N]
    avgsub_Y_Tensor = y - tf.reduce_mean(y, axis = 1, keepdims=True);  #[L, N]

    sumed_Pow_X_Tensor = tf.reduce_sum(tf.pow(avgsub_X_Tensor, 2), axis=1, keepdims= True)      #[M, 1]
    sumed_Pow_Y_Tensor = tf.reduce_sum(tf.pow(avgsub_Y_Tensor, 2), axis=1, keepdims= True)    #[L, 1]

    correlation_Tensor = tf.matmul(avgsub_X_Tensor, tf.transpose(avgsub_Y_Tensor)) / tf.sqrt(tf.matmul(sumed_Pow_X_Tensor, tf.transpose(sumed_Pow_Y_Tensor)));    #[M, L]
    p_Value_Tensor = 1 - tf.erf(tf.abs(correlation_Tensor) * tf.sqrt(tf.cast(tf.shape(x)[1], tf.float32)) / tf.sqrt(2.0));  #[M, L]

    correlation_Tensor = tf.identity(correlation_Tensor, name="correlation");
    p_Value_Tensor = tf.identity(p_Value_Tensor, name="p_value");

    return (correlation_Tensor, p_Value_Tensor)


def Batch_Correlation2D(x, y):
    """
    Compute the correlations between each rows of two tensors. Main purpose is checking the
        correlations between the units of two layers
    Args:
        x: 3d tensor (BATCHxMxN). The number of first and third dimension should be same to y's first and third dimension.
        y: 3d tensor (BATCHxLxN). The number of first and third dimension should be same to x's first and third dimension.
    Returns:        
        correlation_Tensor: A `Tensor` representing the correlation between the rows. Size is (BATCH x M x L)
        p_Value_Tensor: A `Tensor` representing the p-value of correlation. Size is (BATCH x M x L)
    """
    avgsub_X_Tensor = x - tf.reduce_mean(x, axis = 2, keepdims=True);  #[Batch, M, N]
    avgsub_Y_Tensor = y - tf.reduce_mean(y, axis = 2, keepdims=True);  #[Batch, L, N]

    sumed_Pow_X_Tensor = tf.reduce_sum(tf.pow(avgsub_X_Tensor, 2), axis=2, keepdims= True)      #[Batch, M, 1]
    sumed_Pow_Y_Tensor = tf.reduce_sum(tf.pow(avgsub_Y_Tensor, 2), axis=2, keepdims= True)    #[Batch, L, 1]

    correlation_Tensor = tf.matmul(avgsub_X_Tensor, tf.transpose(avgsub_Y_Tensor, perm=[0, 2, 1])) / tf.sqrt(tf.matmul(sumed_Pow_X_Tensor, tf.transpose(sumed_Pow_Y_Tensor, perm=[0, 2, 1])));    #[Batch, M, L]
    p_Value_Tensor = 1 - tf.erf(tf.abs(correlation_Tensor) * tf.sqrt(tf.cast(tf.shape(x)[2], tf.float32)) / tf.sqrt(2.0));  #[M, L]

    correlation_Tensor = tf.identity(correlation_Tensor, name="correlation");
    p_Value_Tensor = tf.identity(p_Value_Tensor, name="p_value");

    return (correlation_Tensor, p_Value_Tensor)
