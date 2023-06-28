import numpy as np
import time
def lscp_greedy(prob_maps, num_prob_maps, A):
    M = np.size(A, 1)
    subsets = []
    for i in range(M):
        index_array = np.where(A[:, i] == 1)
        subsets.append(set(index_array[0].tolist()))
    min_cost = M
    for pmap in range(num_prob_maps):
        #_out = prob_maps[:, pmap]
        _out = prob_maps[:, pmap].cpu().detach().numpy()
        _sorted = np.flip(np.argsort(_out))
        covered = set()
        covering = {}
        solution = np.full(M, -1)
        for v in _sorted:
            if subsets[v].issubset(covered):
                solution[v] = 0
            else:
                solution[v] = 1
                covered.update(subsets[v])
                covering[v] = subsets[v]
        refine_covering = refine(covering)
        solution = [k for k in refine_covering.keys()]
        if min_cost > len(solution):
            min_cost = len(solution)
    #print(sorted(solution))
    return min_cost

def refine(covering):
    nodes = sorted(covering.keys(),key=lambda k:len(covering[k]))
    removed_node = []
    for i in nodes:
        node_cover = covering[i]
        for j in nodes:
            if i == j or j in removed_node:
                continue
            node_cover_by_others = covering[j] & node_cover
            node_cover = node_cover - node_cover_by_others
            if not node_cover:
                break
        if not node_cover:
            removed_node.append(i)

    for rn in removed_node:
        covering.pop(rn)
    return covering


def mclp_greedy(prob_maps, num_prob_maps, A,label):
    M = np.size(A, 1)
    subsets = []
    p = sum(label)
    for i in range(M):
        index_array = np.where(A[:, i] == 1)
        subsets.append(set(index_array[0].tolist()))
    print(subsets)
    label_coverd = set()
    for index in range(len(label)):
        if label[index]==1:
            label_coverd.update(subsets[index])
    max_covering = 0
    # min_solution = None
    for pmap in range(num_prob_maps):
        _out = prob_maps[:, pmap].cpu().detach().numpy()
        _sorted = np.flip(np.argsort(_out))
        covered = set()
        solution = np.full(M, -1)
        for v in _sorted:
            if len(np.where(solution==1)[0]) == p or subsets[v].issubset(covered):
                solution[v] = 0
            else:
                solution[v] = 1
                covered.update(subsets[v])
        if max_covering < len(covered):
            max_covering = len(covered)
            max_solution = solution
    #print(np.where(max_solution==1))
    return max_covering,len(label_coverd)

if __name__ == '__main__':
    M = 10
    N = 10
    num_prob_maps = 3
    prob_maps = np.random.rand(M, num_prob_maps)
    A = np.zeros((N, M))

    for i in range(N):
        A[i, np.random.choice(M, size=np.random.randint(1, M), replace=False)] = 1
    print(A)
    a_time = time.time()
    min_cost = lscp_greedy(prob_maps,num_prob_maps,A)
    print(min_cost,time.time()-a_time)