'''Post-processing
Registration post-processing after registration: 
    After the first round of post-processing with registration as the indicator, obtain the transformed template and its corresponding anatomical landmarks.
    Then, use the transformed template and its landmarks as the new template and landmarks to perform another round of post-processing with registration as the indicator on the validation data.

Purpose: The first round of post-processing is to register the template with the validation data, eliminate large-scale rotational mismatches, and further find a suitable candidate point range.

Therefore, the execution steps are as follows:
1. Set a parameter for registration-only based on the post-processing parameters of registration to return the transformed template and its anatomical landmarks.
2. Build a dataset for the transformed template and anatomical landmarks and extract embeddings.
3. Determine the candidate point range using the transformed template landmarks and select candidate points.
4. Build a dataset for the reselected validation data and its candidate points and extract embeddings.
5. Perform the real post-processing steps.
'''

import numpy as np
import torch
import SimpleITK as sitk
import torch.nn.functional as F
from sklearn.neighbors import LocalOutlierFactor


def PostProcess_registration_wo_dirConsistency(templateArray, validArray, template_afids, candidate_pointSet, var_errTolerance=0, whichInit='LOF_top1'):
    # Establish the initial correspondence
    afid_num = len(template_afids)
    template_afids = np.array(template_afids)
    candidate_pointSet = np.array(candidate_pointSet)

    indexs = initialization(candidate_pointSet)

    if whichInit == "top1":
        last_index = indexs['top1']
    if whichInit == "LOF_top1":
        last_index = indexs['LOF_top1']
    if whichInit == "LOF_medoid":
        last_index = indexs['LOF_medoid']
    if whichInit not in ["source", "top1", 'LOF_top1', 'LOF_medoid']:
        print("WRONG")
        return None
    # init_correspondencePoints = candidate_pointSet[np.arange(afid_num), last_index, :]

    last_error = MI_with(templateArray, validArray)

    Tag_IterationOver = True
    iter_error_list = [last_error]
    iter_index_list = [last_index]

    iteration = int(candidate_pointSet.shape[1] / 2)
    for iter in range(iteration):
        # dist consistency: transform template and afids by distance geometric constraints
        transed_TemplateArray, transed_template_afids = distance_consistency(templateArray, validArray, template_afids, candidate_pointSet, last_index)

        # direction consistency: DO Not use direction consistency
        _, _, distanceResult_Corrpoints, index = direction_consistency(transed_template_afids, candidate_pointSet)

        if iter == 0:  # save the first registration result( template and afids )
            Reged_TemplateArray, Reged_template_afids = transed_TemplateArray, transed_template_afids

        if np.all(np.array(index) == np.array(last_index)):
            Tag_IterationOver = False
            break   # if no change, break

        # calculate the error between the transformed template and the validArray
        error = MI_with(transed_TemplateArray, validArray)

        # calculate the change of error
        var_error = error - last_error 
        # if the change of error is less than Tolerance, select the last result
        if iter != 0:
            if  var_error < var_errTolerance:
                Tag_IterationOver = False
                index = last_index
                break

        # save the result of this iteration
        iter_error_list.append(error)
        iter_index_list.append(index)
        
        # update the variables
        last_index = index
        last_error = error

    # if arrive the max iteration times
    if Tag_IterationOver == True:
        # select the index with the max error
        maxIter = np.argmax(iter_error_list)
        finalIndex = iter_index_list[maxIter]
    else:
        finalIndex = index
    finalCoordinates = candidate_pointSet[np.arange(afid_num), finalIndex, :]
    return finalCoordinates, finalIndex, Reged_TemplateArray, Reged_template_afids

def PostProcess_modify_registration(templateArray, validArray, template_afids, candidate_pointSet, var_errTolerance=0, whichInit='LOF_top1'):
    # Establish the initial correspondence
    afid_num = len(template_afids)
    template_afids = np.array(template_afids)
    candidate_pointSet = np.array(candidate_pointSet)
    
    # Calculate the center coordinates of the candidate set after removing outliers
    indexs = initialization(candidate_pointSet)

    # if whichInit=="source":
    #     last_index = closest_index
    if whichInit == "top1":
        last_index = indexs['top1']
    if whichInit == "LOF_top1":
        last_index = indexs['LOF_top1']
    if whichInit == "LOF_medoid":
        last_index = indexs['LOF_medoid']
    if whichInit not in ["source", "top1", 'LOF_top1', 'LOF_medoid']:
        print("WRONG")
        return None
    init_correspondencePoints = candidate_pointSet[np.arange(afid_num), last_index, :]
    # Calculate the average error of the current points
    # last_error = compute_2Set_MeanDistance(init_correspondencePoints, template_afids)
    last_error = MI_with(templateArray, validArray)
    
    # Enter the loop (loop termination condition: change in error between two iterations)
    Tag_IterationOver = True
    iter_error_list = [last_error]
    iter_index_list = [last_index]

    iteration = int(candidate_pointSet.shape[1] / 2)
    for iter in range(iteration):
        # Transform the coordinates of the template
        transed_TemplateArray, transed_template_afids = distance_consistency(templateArray, validArray, template_afids, candidate_pointSet, last_index)
       
        # Select a new tentative set
        tentativeSet, index, _, _ = direction_consistency(transed_template_afids, candidate_pointSet)
        
        # Convergence condition judgment
        # Transform the template again based on the tentative set obtained from direction and distance consistency constraints
        Final_TemplateArray, Final_template_afids = distance_consistency(transed_TemplateArray, validArray, transed_template_afids, candidate_pointSet, last_index, Final=True, tentativeSet=tentativeSet)
        
        # Check if the indices of the tentative set have changed
        if iter == 0:  # If it's the pre-registration part
            Reged_TemplateArray, Reged_template_afids = Final_TemplateArray, Final_template_afids
        
        if np.all(np.array(index) == np.array(last_index)):
            Tag_IterationOver = False
            break   # If there is no change, terminate the loop
        # Calculate the error between the new tentative set and the template
        error = MI_with(Final_TemplateArray, validArray)
        # Calculate the error change
        var_error = error - last_error 
        # If the average increase in change is 1mm, terminate the loop and select the result of the previous iteration
        if iter != 0:
            if  var_error < var_errTolerance:
                Tag_IterationOver = False
                index = last_index
                # return None
                break

        # Save the record of this iteration
        iter_error_list.append(error)
        iter_index_list.append(index)
        
        # Print iteration information
        
        # Enter the next loop
        last_index = index
        last_error = error

    # If the maximum number of iterations is reached
    if Tag_IterationOver == True:
        # Select the corresponding points with the maximum loss calculation as the corresponding set
        maxIter = np.argmax(iter_error_list)
        # minIter = np.argmin(iter_error_list)
        finalIndex = iter_index_list[maxIter]
        # finalIndex = iter_index_list[minIter]
    else:
        finalIndex = index
    finalCoordinates = candidate_pointSet[np.arange(afid_num), finalIndex, :]
    return finalCoordinates, finalIndex, Reged_TemplateArray, Reged_template_afids

def initialization(candidate_pointSet):
    indexs = {}
    indexs['top1'] = np.zeros(32, dtype=np.int16)
    LOF_top1_index = []
    LOF_medoid_index = []

    lof = LocalOutlierFactor(n_neighbors=3)

    for i, candidate_points in enumerate(candidate_pointSet):
        # Use LOF to remove outliers
        # plot3dScatter(candidate_points, title=f"{i}")
        outlier_labels = lof.fit_predict(candidate_points)
        # Find the indices of outliers based on outlier scores
        inlier_indices = np.where(outlier_labels != -1)
        inlier_points = candidate_points[inlier_indices]
        # print("Outliers are:", np.where(outlier_labels == -1))
        
        LOF_top1_index.append(inlier_indices[0][0])

    indexs['LOF_top1'] = np.array(LOF_top1_index)
    # Calculate the point in the candidate set closest to the sample center and return its index
    return indexs

def distance_consistency(template_array, valid_array, template_afids, candidate_topPoints, CorrIndex, Final=False, tentativeSet=None):
    afid_num = len(template_afids)
    template_afids = np.array(template_afids)
    candidate_topPoints = np.array(candidate_topPoints)
    
    # Select corresponding points according to the index
    if not Final:
        correspondencePoints = candidate_topPoints[np.arange(afid_num), CorrIndex, :] # afid_num * 3
    else:
        assert tentativeSet is not None
        correspondencePoints = tentativeSet
    # Calculate the distance matrix between points within the template and the tentative set (afid_num * afid_num)
    
    scale = compute_scaleFactor_with(template_afids, correspondencePoints)

    # Scale the template data centered using the scaling factor and transform the image once
    scaled_TemplateArray, scaled_template_afids = ImageArray_3DTransformeWithPoint(template_array, template_afids, np.eye(3).astype(np.float32), scale, np.zeros(3))
    
    # Calculate the rotation and translation vectors from the transformed template point set to the tentative set
    Rotation, trans, Rcenter = compute_R_t_fromICP_quaternion(scaled_template_afids, correspondencePoints)
    
    transed_TemplateArray, transed_template_afids = ImageArray_3DTransformeWithPoint(scaled_TemplateArray, scaled_template_afids, Rotation, 1, trans, center=Rcenter)
   
    return transed_TemplateArray, transed_template_afids

def direction_consistency(transed_templatePoints, candidate_topPoints, threshold_z_score=2, ratio_dis=0.5):
    '''Use the template coordinates constrained by distance to select a tentative set with the best distance and direction consistency from the candidate point set.
    '''
    # 1. Calculate the new point set closest to the template point set in the initial condition
    afid_num = len(transed_templatePoints)
    transed_templatePoints = np.array(transed_templatePoints)
    candidate_topPoints = np.array(candidate_topPoints)
    
    dis_t2c = np.linalg.norm(transed_templatePoints[:, np.newaxis] - candidate_topPoints, axis=2) # distance_template_to_candidate
    # Sort the candidate set in ascending order of distance
    indexs_matrix = np.argsort(dis_t2c, axis=1)
    i_index = np.zeros(shape=(afid_num,), dtype=np.int32)
    # 2. Calculate the angular error between the new point set and the template point set and find outliers
    temp_set_index = indexs_matrix[np.arange(afid_num), i_index]
    candidate_tempPoints = candidate_topPoints[np.arange(afid_num), temp_set_index] # afid_num * 3
    distanceResult_Corrpoints, distanceResult_Indexs = candidate_tempPoints.copy(), temp_set_index.copy()

    error_z_scores = dirConsis_2Set_Zscore(transed_templatePoints, candidate_tempPoints)
    outliers = np.abs(error_z_scores) > threshold_z_score

    if np.all(outliers == False):
        # If there are no directional outliers
        # print(f"No directional outliers in the {i+1}th point selection.")
        return candidate_tempPoints, temp_set_index, distanceResult_Corrpoints, distanceResult_Indexs
        
    # 3. Calculate the weighted error of distance and direction for 20 points in the candidate set of outliers
    # This calculation needs to be performed for each outlier
    for i, outlier in enumerate(outliers):
        if not outlier:
            continue
        # First, normalize the distance between the candidate point set and the template point to get the distance error.
        # The distance of the candidate set where the outlier is located.
        outlier_candidates_dis = dis_t2c[i, :]
        norm_dis_Errors = norm_array(outlier_candidates_dis)
        
        # Replace the outlier with all points in the candidate set, calculate the directional error (z-score), and then normalize it.
        outlier_candidates = candidate_topPoints[i, :]
        dir_ZScores = []
        for j, o_candi in enumerate(outlier_candidates):
            o_candi_candidate_tempPoints = candidate_tempPoints.copy()
            o_candi_candidate_tempPoints[i, :] = o_candi
            o_candi_z_scores = dirConsis_2Set_Zscore(transed_templatePoints, o_candi_candidate_tempPoints)
            dir_ZScores.append(o_candi_z_scores[i])
        norm_dir_ZScores = norm_array(dir_ZScores)
        
        # Weighted sum of distance and direction errors to find the point with the minimum error to replace the outlier
        weight_error = norm_dis_Errors * ratio_dis + norm_dir_ZScores * (1 - ratio_dis)
        # print(weight_error)
        replace_i = np.argmin(weight_error)
       
        temp_set_index[i] = replace_i
        candidate_tempPoints = candidate_topPoints[np.arange(afid_num), temp_set_index]
        
    return candidate_tempPoints, temp_set_index, distanceResult_Corrpoints, distanceResult_Indexs

def dirConsis_2Set_Zscore(TemplateSet, tentativeSet):
    afid_num = len(TemplateSet)
    DirUnit_Temp = unitDirectionMatrix_PointSet(TemplateSet)
    DirUnit_tent = unitDirectionMatrix_PointSet(tentativeSet)
    error_theta_matrix = np.arccos(np.sum(DirUnit_Temp * DirUnit_tent, axis=2))
    # Perform outlier determination
    np.fill_diagonal(error_theta_matrix, 0)
    np.nan_to_num(error_theta_matrix, copy=False)
    # Sum and average in one dimension and then calculate the z-score
    errorSum_theta = np.sum(error_theta_matrix, axis=1) / (afid_num - 1) # afid_num
    error_z_scores = (errorSum_theta - np.mean(errorSum_theta)) / np.std(errorSum_theta)
    return error_z_scores

def unitDirectionMatrix_PointSet(pointSet):
    '''Calculate the direction vector matrix between points within the point set.
    '''
    direction_pointSetInnerMatrix = pointSet[:, np.newaxis] - pointSet  # afid_num * afid_num * 3
    # Calculate the distance between points within the template
    dis_pointSetInnerMatrix = np.linalg.norm(direction_pointSetInnerMatrix, axis=2)
    np.fill_diagonal(dis_pointSetInnerMatrix, 1)
    # Calculate the unitized direction vector matrix
    directionUnit_pointSetInnerMatrix = direction_pointSetInnerMatrix / dis_pointSetInnerMatrix[:, :, np.newaxis]
    return directionUnit_pointSetInnerMatrix

def MI_with(image1, image2, bins=256, IsArray=True):
    if not IsArray:
        img1 = np.load(image1)
        img2 = np.load(image2)
    else:
        img1 = image1
        img2 = image2
    # Flatten the 3D image data into a 1D array
    flat_img1 = img1.flatten()  
    flat_img2 = img2.flatten()  
    
    hist1, _ = np.histogram(img1, bins=bins, density=True)  
    hist2, _ = np.histogram(img2, bins=bins, density=True)  

    # Calculate the joint histogram
    # Use np.histogram2d to calculate the joint histogram of two 1D arrays
    joint_hist, _, _ = np.histogram2d(flat_img1, flat_img2, bins=bins, density=True)  
    px = hist1  
    py = hist2  
    pxy = joint_hist  
    
    # Calculate the mutual information
    mi = calculate_mutual_information(px, py, pxy)  
    
    # print("Mutual information:", mi)
    return mi

def calculate_entropy(probabilities):  
    """  
    Calculate the entropy of a given probability distribution.  
    
    Parameters:  
        probabilities (np.ndarray): 1D array representing the probability distribution.  
    
    Returns:  
        entropy (float): Entropy value.  
    """  
    # Avoid taking the logarithm of 0
    probabilities = probabilities + np.finfo(float).eps  
    # Calculate the entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))  
    return entropy  

def calculate_mutual_information(px, py, pxy):  
    """  
    Calculate the mutual information between two random variables.  
    
    Parameters:  
        px (np.ndarray): Marginal probability distribution of random variable X.  
        py (np.ndarray): Marginal probability distribution of random variable Y.  
        pxy (np.ndarray): Joint probability distribution of random variables X and Y.  
    
    Returns:  
        mi (float): Mutual information value.  
    """  
    # Calculate the marginal entropies
    hx = calculate_entropy(px)  
    hy = calculate_entropy(py)  
    # Calculate the joint entropy
    hxy = calculate_entropy(pxy.flatten())  
    # Calculate the mutual information
    # print("Hx:", hx)
    # print("Hy:",  hy)
    # print("HXY:", hxy )
    mi = hx + hy - hxy  
    return mi  

def compute_R_t_fromICP_quaternion(template_pointSet, tentative_pointSet):
    '''Input: Template point set and tentative point set, number of iterations, and convergence threshold for the final transformation.
        Output:
            Rotation matrix 3x3,
            Translation vector 1x3,
            Center position of the point set 1x3, used as the rotation center of the image.
    '''
    # The algorithm uses the ICP algorithm for several iterations
    # The ICP algorithm fits the rotation and translation matrix between the scaled template and the tentative set
    tentative_pointSet = np.array(tentative_pointSet)
    template_pointSet = np.array(template_pointSet)
    afid_num = tentative_pointSet.shape[0]
    # Calculate the covariance matrix
    tentative_mean = np.mean(tentative_pointSet, axis=0)
    template_mean = np.mean(template_pointSet, axis=0)
    covarianceMatrix = np.dot((template_pointSet - template_mean).T, (tentative_pointSet - tentative_mean)) / afid_num
    # Calculate the Q matrix
    AMatrix = covarianceMatrix - covarianceMatrix.T
    iMatric = covarianceMatrix + covarianceMatrix.T - np.eye(3) * covarianceMatrix.trace() # intermediateMatric
    QMatrix = np.array([
        [covarianceMatrix.trace(), AMatrix[1,2], AMatrix[2,0], AMatrix[0,1]],
        [AMatrix[1,2], iMatric[0,0], iMatric[0,1], iMatric[0,2]],
        [AMatrix[2,0], iMatric[1,0], iMatric[1,1], iMatric[1,2]],
        [AMatrix[0,1], iMatric[2,0], iMatric[2,1], iMatric[2,2]]
    ])
    # Perform SVD decomposition on the Q matrix
    eigenvalues, eigenvectors = np.linalg.eigh(QMatrix)  
    # Calculate the eigenvector corresponding to the maximum eigenvalue of the Q matrix
    q = eigenvectors[:, -1]  # qVector
    # Calculate the rotation matrix R according to the formula
    RotationMatrix = np.array([
        [q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2, 2*(q[1] * q[2] - q[0] * q[3]), 2*(q[1] * q[3] + q[0] * q[2])],
        [2*(q[1] * q[2] + q[0] * q[3]), q[0]**2 + q[2]**2 - q[1]**2 - q[3]**2, 2*(q[2] * q[3] - q[0] * q[1])],
        [2*(q[1] * q[3] - q[0] * q[2]), 2*(q[2] * q[3] + q[0] * q[1]), q[0]**2 + q[3]**2 - q[1]**2 - q[2]**2]
    ])
    # Calculate the translation parameter based on the distance between the centers of the two point sets
    TranslationVector = tentative_mean - template_mean
    # TranslationVector = tentative_mean - np.dot(RotationMatrix, template_mean) 
    # TranslationVector = get_CenterOfMass(validArray) 
    # Finally, transform the template point set
    # Save the previous one
    # transform_template_pointSet = np.dot(RotationMatrix, template_pointSet.T).T + TranslationVector
    # return transform_template_pointSet
    return RotationMatrix, TranslationVector, template_mean

def compute_scaleFactor_with(pointSet1, pointSet2, z_TH=2):
    '''
    Estimate the scaling factor from set1 to set2 based on the correspondence.
    '''
    # Calculate the distance matrix between points within the template and the tentative set (afid_num * afid_num)
    dis_matrix1 = compute_PointSet_distanceMatrix(pointSet1)
    dis_matrix2 = compute_PointSet_distanceMatrix(pointSet2)

    # Divide the two distance matrices, remove the diagonal elements, and calculate the scaling factor
    candidate_ScaleMatrix = dis_matrix2 / dis_matrix1
    candidate_ScaleMatrix_dropDiagElement = candidate_ScaleMatrix[~np.eye(candidate_ScaleMatrix.shape[0], dtype=bool)]
    
    candidate_ScaleMatrix_z_score = Z_score(candidate_ScaleMatrix_dropDiagElement)
    outliers = np.abs(candidate_ScaleMatrix_z_score) > z_TH
    candidate_ScaleMatrix_dropDiagElement_dropexception = candidate_ScaleMatrix_dropDiagElement[~outliers]
    # Calculate the mean and keep two decimal places
    scale_byMean = np.round(np.mean(candidate_ScaleMatrix_dropDiagElement_dropexception), 2)

    return scale_byMean

def compute_PointSet_distanceMatrix(pointSet, FillDiagonal=True):
    pointSet = np.array(pointSet)
    dis_Matrix = np.linalg.norm(pointSet[:, np.newaxis] - pointSet, axis=2)
    if FillDiagonal:
        np.fill_diagonal(dis_Matrix, 1)
    return dis_Matrix

def Z_score(array1d):
    z_score = (array1d - np.mean(array1d)) / np.std(array1d)
    return z_score

def norm_array(x):
    x = np.array(x)
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def get_CenterOfMass(array):
    valid_indexs = np.where(array > 0)
    center_of_mass = np.array(valid_indexs).mean(axis=1)
    return center_of_mass

def ImageArray_3DTransformeWithPoint(nparray, points, RotationScale=np.eye(3), scale=1, trans=np.zeros(3), center=None):
    '''
    Perform corresponding transformations (rotation, scaling, translation) on the image and its points.
    Input:
        nparray: H, W, D
        points: n, 3
        RotationScale: 3, 3
        scale: 1
        trans: 1, 3

    1. Generate a rotation and scaling matrix.
    2. Based on torch.nn.functional.grid, use the rotation and scaling matrix to transform the image and points.
    3. Generate a translation vector.
    4. Based on SimpleITK, generate a translation transformation and use sitk.Resample to translate the image.

    Note 2: The storage of arrays in numpy is (z, y, x), so the numpy array needs to be transposed first and then sent to torch or sitk for transformation processing.
    Note 3: After converting to numpy, it needs to be converted to (x, y, z).
    Note 4: The translation in sitk may be opposite in the Depth axis direction for points and images due to the direction problem.
    Note 5: The scaling in sitk also needs to be opposite for the image and points (to be tested).
    Rotation: Counterclockwise is the positive direction of rotation.
    Note 1: The points need to be transformed simultaneously, but it should be noted that the transformation matrix of points is opposite to the direction of the grid in torch, so the transformations of the two are transpose matrices of each other. See 1.
    '''
    # Basic settings of the image
    if center is None:
        center = get_CenterOfMass(nparray)
    ras_direction = [-1, 0, 0, 0, -1, 0, 0, 0, 1]  # Default direction of the image represented by the array. In SITK, it is LPS by default, so the RAS direction needs to be set.
    
    # Perform a rotation transformation on the image
    nparray = nparray.transpose(2,1,0).astype(np.float32)  # Note 2
    tensor_array = torch.tensor(nparray).unsqueeze(0).unsqueeze(0) 
    size = [1, 1, *nparray.shape]
    RotationScale_M = np.hstack([RotationScale, np.zeros((3,1))]).astype(np.float32)
    RotationScale_M = np.vstack([RotationScale_M, np.array([0, 0, 0, 1])]).astype(np.float32)
    RotationScale_M = RotationScale_M.T  # Note 1
    assert RotationScale_M.shape == (4, 4)
    grid = F.affine_grid(torch.tensor(RotationScale_M[0:3]).unsqueeze(0), size, align_corners=True) # matrix.shape=(1,3,4), size=(N, C, W, H, D)
    rotationScale_array = F.grid_sample(tensor_array, grid, mode='bilinear', align_corners=True)
    rotationScale_array = rotationScale_array.numpy().squeeze()
    rotationScale_array = rotationScale_array.transpose(2,1,0)  # Note 3
    
    # Perform a rotation transformation and scaling on the points
    rot_offset = np.array(center) - RotationScale / (2 - scale) * scale @ np.array(center).T 
    rotationScale_points = (RotationScale / (2 - scale) * scale @ points.T).T + rot_offset

    # Perform a translation/scaling transformation on the image (after rotation)
    sitk_image = sitk.GetImageFromArray(rotationScale_array.transpose(2,1,0))  # Note 2
    sitk_image.SetDirection(ras_direction)

    # scale_matrix = np.eye(3) * (1 + 1 - scale)
    # print(trans)
    trans = trans * np.array([1,1,-1])
    translate_trans = sitk.AffineTransform(np.eye(3).flatten().astype(np.float32).tolist(), trans.tolist())
    RotScaleTranslate_image = sitk.Resample(
        sitk_image, 
        translate_trans
    )
    sitkArray = sitk.GetArrayFromImage(RotScaleTranslate_image)
    sitkArray = sitkArray.transpose(2, 1, 0)
    # Perform a translation on the points
    trans_offset =  trans * np.array([1,1,-1]) # Note 4, Note 5
    RotScaleTranslate_points =  rotationScale_points + trans_offset                    

    return sitkArray, RotScaleTranslate_points

