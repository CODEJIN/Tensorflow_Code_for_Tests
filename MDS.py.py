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

    j = tf.eye(element_Number) - 1/element_Number * tf.ones_like(x);
    b = 0.5 * (j @ tf.pow(x, 2) @ j);
    eigen_Value, eigen_Vector = tf.self_adjoint_eig(b)
    mds_Coordinate = eigen_Vector[:, :dimension] @ tf.sqrt(tf.diag(-eigen_Value[:dimension]));
    mds_Coordinate = tf.identity(mds_Coordinate, name="mds_Coordinate");

    return mds_Coordinate;