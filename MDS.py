import tensorflow as tf;

def MDS(x, dimension = 2):
    """
    Compute the multidimensional scaling coordinates.
    Equation reference: https://m.blog.naver.com/PostView.nhn?blogId=kmkim1222&logNo=220082090874&proxyReferer=https%3A%2F%2Fwww.google.com%2F
    Args:
        x: 2d tensor (NxN). The distance matrix.
        dimension: int32 or scalar tensor. The compressed dimension.
    Returns:        
        mds_Coordinate: A `Tensor` representing the compressed coordinates. Size is (N x Dimension)
    """
    element_Number = tf.shape(x)[1];    
    j = tf.eye(element_Number) - tf.cast(1/element_Number, tf.float32) * tf.ones_like(x);
    b = -0.5 * (j @ tf.pow(x, 2) @ j);
    eigen_Value, eigen_Vector = tf.self_adjoint_eig(b)
    selected_Eigen_Value, top_Eigen_Value_Indice = tf.nn.top_k(eigen_Value, k=dimension);
    selected_eigen_Vector = tf.transpose(tf.gather(tf.transpose(eigen_Vector), top_Eigen_Value_Indice))
    mds_Coordinate = selected_eigen_Vector @ tf.sqrt(tf.diag(selected_Eigen_Value));
    mds_Coordinate = tf.identity(mds_Coordinate, name="mds_Coordinate");

    return mds_Coordinate;
