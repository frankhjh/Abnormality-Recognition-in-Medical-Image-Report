#deal with label
def label_transform(prob_info,dim=17):
    label_vector=[0 for i in range(dim)]
    if prob_info=='-':
        return label_vector
    else:
        prob_regions=list(map(lambda x:int(x),prob_info.strip().split()))
        for i in prob_regions:
            label_vector[i]=1
    return label_vector    