import torch
import numpy as np
from torch.optim.optimizer import Optimizer
import copy

class MultiQueue():
    """
        Queue class that allows queue manipulation with respect to given que key, que length ordered pairs.
        All queues are set to their que lengths, appending a value to a queue pops the 0th element. That is queue
        lengths are kept constant.

        E.g.:
        que_key_lenght_ordered_pairs={'a':5, 'b':2, 'c':3}
        m = Multiqueue (que_key_lenght_ordered_pairs)

        for m, a list "q" is created where first 5 entries are associated with queue "a",
        next 2 are associated with queue "b" and the next 3 are associated with queue "c"

        access the list by m.q

    """
    def __init__(self,queue_keys_bounding_length_pairs):
        self.queue_info = dict()
        self.key2index = dict()

        self.q=[] #the list that holds all the queues elements
        start_index=0

        # set partition indices for each queue, and set each element as 'EMPTY'
        for key in queue_keys_bounding_length_pairs:

            self.queue_info[key]={'start':start_index,
                                    'end':start_index + queue_keys_bounding_length_pairs[key] -1,
                                    'length':0}
            start_index=start_index + queue_keys_bounding_length_pairs[key]

            for i in range(queue_keys_bounding_length_pairs[key]):
                self.q.append('EMPTY')


    def appnd(self,key,val):
        """
        method to append values to specified queues
        :param key: queue key to access the relevant part of q
        :param val: value to append
        :return:

         E.g.:
        que_key_lenght_ordered_pairs={'a':3, 'b':2}
        m = Multiqueue (que_key_lenght_ordered_pairs)
        print(m.q) >>> ['EMPTY','EMPTY','EMPTY','EMPTY','EMPTY']
        m.appnd('a',1)
        print(m.q) >>> [1,'EMPTY','EMPTY','EMPTY','EMPTY']
        m.appnd('a',2)
        print(m.q) >>> [1,2,'EMPTY','EMPTY','EMPTY']
        m.appnd('b',10)
        print(m.q) >>> [1,2,'EMPTY',10,'EMPTY']
        m.appnd('a',3)
        print(m.q) >>> [1, 2, 3, 10,'EMPTY']
        m.appnd('a',4)s
        print(m.q) >>> [2, 3, 4, 10,'EMPTY']
        m.appnd('b',20)
        print(m.q) >>> [2, 3, 4, 10, 20]
        """

        # if queue is at its full length pop from the start of the queue and append the value the last
        if(self.queue_info[key]['length']==(self.queue_info[key]['end']-self.queue_info[key]['start'])+1):
            self.q.pop(self.queue_info[key]['start'])
            self.q.insert(self.queue_info[key]['end'],val)

        # if queue is not at its full length pop from the start of the queue and append the value the last,
        # also update the queue length
        else:
            self.q.pop(self.queue_info[key]['end'])
            self.q.insert(self.queue_info[key]['start']+self.queue_info[key]['length'],val)
            self.queue_info[key]['length']=self.queue_info[key]['length']+1

    def get(self,key,pos):
        """

        :param key: queue key to access the relevant part of q
        :param pos: elements to get from the queue specified by the key
        :return: queue elements (list for #elements>1, element for #elements=1 )

        E.g.:
        que_key_lenght_ordered_pairs={'a':3, 'b':2}
        m = Multiqueue (que_key_lenght_ordered_pairs)
            ...
        print(m.q) >>> [2, 3, 4, 10, 20]
        print(m.get('a',(0,2))) >>> [2, 3]
        print(m.get('b',(0,1))) >>> 10

        """
        if (pos[1]-pos[0])>1:
            return self.q[pos[0]+self.queue_info[key]['start']:pos[1]+self.queue_info[key]['start']]
        elif (pos[1]-pos[0])==1:
            return self.q[pos[0] + self.queue_info[key]['start']:pos[1] + self.queue_info[key]['start']][0]
        else:
            return

    def get_last(self,key):
        """

        :param key: queue key to access the relevant part of q
        :return: last element of the specified queue

        E.g.:
        que_key_lenght_ordered_pairs={'a':3, 'b':2}
        m = Multiqueue (que_key_lenght_ordered_pairs)
            ...
        print(m.q) >>> [2, 3, 4, 10, 20]
        print(m.get_last('a') >>> 4
        print(m.get_last('b') >>> 20
        """
        return self.get(key,[self.queue_info[key]['length']-1,self.queue_info[key]['length']])

class SymmetricMatrix():
    """
        Holds m by m symmetric matrix using its upper diagonal entries as an array. Instead of mxm values stores ~mxm/2
    """
    def __init__(self,size):
        self.m=size
        self.arr_size= int((size*size-size)/2 + size)
        self.values=np.zeros(shape=(self.arr_size,1))

    def _rc2indx(self,row,col):

        if(row>col):
            r=col
            c=row
        else:
            r=row
            c=col

        return int((self.m*r+r-((r*(r+1))/2))*(r>0) + (c-r))


    def set(self,coordinate,value):
        """
        Set the value at ith row and jth coloumn to "value"
        :param coordinate: (i,j)
        :param value: value to set
        :return:
        """
        self.values[self._rc2indx(coordinate[0],coordinate[1])]=value

    def get(self,coordinate):
        """
        Get the value at ith row and jth coloumn
        :param coordinate: (i,j)
        :return: value
        """
        return (self.values[self._rc2indx(coordinate[0],coordinate[1])][0])

    def get_subset(self,start_coord,subset_shape):
        """
        Get the subset that starts from the start_coord and has a shape of subset_shape
        :param start_coord: (i,j)
        :param subset_shape:  (y_length,x_length)
        :return: 2D numpy array
        """
        subset_2D=np.zeros(shape=(subset_shape[0],subset_shape[1]))

        for i in range(subset_shape[0]):
            for j in range(subset_shape[1]):
                subset_2D[i, j] = self.values[self._rc2indx(start_coord[0]+i, start_coord[1]+j)]

        return subset_2D

    def set_subset(self, start_coord, subset_2D):
        """
        Replace the subset that starts from the start_coord with subset_2D
        :param start_coord: (i,j)
        :param subset_2D: 2D numpy array
        :return:
        """
        subset_2D=np.asarray(subset_2D)
        subset_shape=subset_2D.shape

        for i in np.arange(start_coord[0],start_coord[0]+subset_shape[0]):
            for j in np.arange(start_coord[1],start_coord[1]+subset_shape[1]):
                self.values[self._rc2indx(i, j)]=subset_2D[i-start_coord[0],j-start_coord[1]]



def Armijo_Rule(f_next,f_initial,c1,step_size,pg_initial):
    """
    :param f_next: New value of the function to be optimized wrt/ step size
    :param f_initial: Value of the function before line search for optimum step size
    :param c1: 0<c1<c2<1
    :param step_size: step size to be tested
    :param pg: inner product of step direction, p, with the gradient before stepping, g_initial
    :return: True if condition is satisfied
    """
    return (f_next <= f_initial+c1*step_size*pg_initial)

def Cuvature_Condition(pg_next,c2,pg_initial):
    """
    :param pg_next: inner product of step direction, p, with the gradient after stepping, g_next
    :param c2: 0<c1<c2<1
    :param pg_initial: inner product of step direction, p, with the gradient before stepping, g_initial
    :return:True if condition is satisfied
    """
    return  (-pg_next <= -c2*pg_initial)


class VLBFGS(Optimizer):
    """
    LBFGS Optimizer with multiprocessing option on CPU.
    (VLBFGS: Vector Free LBFGS https://papers.nips.cc/paper/2014/hash/e49b8b4053df9505e1f48c3a701c0682-Abstract.html)

    ------------------------------------------------------------------
    E.g.:
    from pathos.multiprocessing import ProcessingPool as Pool

    ...

    pool = Pool()
    optimizer = VLBFGS(params=model.parameters(), \
                       step_size=1, m=10, max_iteration=100, termination_threshold=0.01, \
                       check_Wolfie=True, c1=1e-4, c2=0.1, \
                       pool=pool, cpu_count_for_pool=mp.cpu_count(), parallel_superposition=True, parallel_innerproduct=True)

    for input_x, output_y in dataset_xy:
        def closure():
            optimizer.zero_grad()
            output_est = model(input_x)
            loss = loss_fn(output_est, output_y)
            loss.backward()
            return loss

        loss = optimizer.step(closure)

    ------------------------------------------------------------------
        IMPORTANT NOTES:

    --> step_size parameter is only needed when Wolfie conditions are not needed to be satisfied.

    --> History size for step direction calculation, m,
     m should be smaller than maximum iterations during a single step, max_iteration.

    --> Set termination_threshold to stop optimization after a certain loss is achieved. If termination_threshold
    is set to 0 optimizer may become unstable.

        Wolfie Conditions:
    --> Set check_Wolfie to True if Wolfie Conditions are needed to be satisfied during optimization. Otherwise step size
    is fixed to step_size. If check_Wolfie is set to True, step_size does not have to be declared during initialization.

    --> Set c1 and c2 accordingly 1>c2>c1>0 (see Armijo rule and curvature https://en.wikipedia.org/wiki/Wolfe_conditions)

    --> line_search_max_iterations is the total iterations during line search to satisfy Weak Wolfie Conditions,
    If line search iterations reach to this value, optimizer.step() stops at the smallest loss found


        Parallel Processing:
    --> For parallel processing with multiple CPU's parallel_superposition and/or parallel_innerproduct should be set to True
    If any of them is set to True, pool must be passed, if not pool is not necessary.

    --> from pathos.multiprocessing import ProcessingPool as Pool and pass it to the "pool" parameter of the optimizer.

    --> cpu_count_for_pool is the cpu count for pooling (map-reduce steps for inner product calculations and tensor additions)

    --> see Figure 1 from
                https://papers.nips.cc/paper/2014/hash/e49b8b4053df9505e1f48c3a701c0682-Abstract.html
            for the number of nn parameters that yield a better performance wrt/ LBFGS
    """

    def __init__(self,params,\
                 step_size=1, m=10, max_iteration=100, termination_threshold=1e-1, \
                 check_Wolfie=True, c1=1e-4, c2=0.1, line_search_max_iterations=50,\
                 pool = None, cpu_count_for_pool=1 ,parallel_superposition=False, parallel_innerproduct=False):
        """

        :param params: nn parameters
        :param step_size: Step size for optimization without checking Wolfie Cond.
        :param m: history size
        :param max_iteration: otal optimization iterations
        :param termination_threshold: rmination threshold for loss
        :param check_Wolfie: If set to True, step size is adjusted by checking Wolfie Conditions (line search)
        :param c1: c1 parameter for Wolfie Condition #1
        :param c2: c2 parameter for Wolfie Condition #2
        :param line_search_max_iterations: Maximum number of iterations to perform during line search
        :param pool: athos.multiprocessing.ProcessingPool
        :param cpu_count_for_pool: number of CPUs for map reduce
        :param parallel_superposition: enable/disable parallel execution of superposition step
        :param parallel_innerproduct: enable/disable parallel execution of innerproduct step
        """

        defaults=dict(m=m,max_iteration=max_iteration,step_size=step_size, termination_threshold=termination_threshold, \
                      check_Wolfie=check_Wolfie,c1=c1,c2=c2,line_search_max_iterations=line_search_max_iterations,\
                      parallel_superposition=parallel_superposition,parallel_innerproduct=parallel_innerproduct)
        super(VLBFGS,self).__init__(params,defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        if c1>c2 or c1<0 or c2>1:
            raise ValueError("Select c1 and c2 satisfying 1>c2>c1>0 ")

        if (parallel_superposition or parallel_innerproduct) and (pool is None):
            raise ValueError("pool should be set from pathos.multiprocessing.ProcessingPool")

        self._params = self.param_groups[0]['params']

        # set b vectors for VLBFGS as multiqueue
        # k = current iteration
        # for 0<=i<m b_(i) = s_(k-m+i)
        # for m<=i<2m b_(i) = y_(k-m-i)
        # for i=2m    b_(i) = g_(k)

        self.state['b']=MultiQueue({'s':m,
                                    'y':m,
                                    'g':1})

        # correlation matrix as symmetricmatrix
        self.state['R']=SymmetricMatrix(2*m+1)
        self.state['iteration']=0
        self.state['loss']=np.Inf

        self.pool = pool
        self.nof_cpu=cpu_count_for_pool

        # set inner product and superposition partition indices for multiple CPUs
        if pool is not None:
            self.partition_starts,self.nof_products_per_cpu = self._get_parallel_partition_indicies()



    def _get_parallel_partition_indicies(self):
        """
        Get paralLel partition indices by setting partitions equally for each CPU
        :return: partition_starts (starting indices of b vectors for each CPU), nof_products_per_cpu (# vectors for each CPU)
        """
        nof_cpu = self.nof_cpu
        m=self.defaults['m']

        nof_products_per_cpu=np.zeros(shape=(nof_cpu))+np.floor((2*m+1)/nof_cpu)

        for i in range(np.mod(2*m+1,nof_cpu)):
            nof_products_per_cpu[i]=nof_products_per_cpu[i]+1

        partition_starts= np.zeros(shape=(nof_cpu),dtype=int)

        for i in range(1,nof_cpu,1):
            partition_starts[i]=partition_starts[i-1]+nof_products_per_cpu[i-1]

        partition_starts = partition_starts.astype(int)
        nof_products_per_cpu = nof_products_per_cpu.astype(int)

        return partition_starts,nof_products_per_cpu

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is not None:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)


    def _update_R(self,s_all,y_all,g_all):
        """
        Updates R after inner product calculations are completed
        :param s_all: inner product of s_(k-1) with b vectors
        :param y_all: inner product of y_(k-1) with b vectors
        :param g_all: inner product of g_(k) with b vectors
        :return:
        """
        m=self.defaults['m']

        ss=self.state['R'].get_subset([1,1],[m-1,m-1])
        ys=self.state['R'].get_subset([1,m+1],[m-1,m-1])
        yy=self.state['R'].get_subset([m+1,m+1],[m-1,m-1])

        self.state['R'].set_subset([0, 0], ss)
        self.state['R'].set_subset([0, m], ys)
        self.state['R'].set_subset([m, m], yy)

        self.state['R'].set_subset([0, m - 1],s_all)
        self.state['R'].set_subset([0, 2*m - 1], y_all)
        self.state['R'].set_subset([0, 2*m ], g_all)

    def _calculate_d(self):
        """
        Vectorized implementation of calculation of d as described
                in https://papers.nips.cc/paper/2014/hash/e49b8b4053df9505e1f48c3a701c0682-Abstract.html
        :return: d
        """
        m = self.defaults['m']
        d=np.zeros(shape=(2*m+1,1))
        d[-1]=-1

        alpha=np.zeros(shape=(m,1))

        for j in range(m-1,-1,-1):
            alpha[j] = np.einsum('ij,ij',d , self.state['R'].get_subset((0,j),(2*m+1,1)))

            r=self.state['R'].get((j,j+m))
            if r!=0:
                alpha[j] = alpha[j] / r

            d[m+j] = d[m+j] - alpha[j]

        r=self.state['R'].get((2*m-1,2*m-1))
        if r !=0:
            d=d*self.state['R'].get((m-1,2*m-1))/r

        for j in range(0, m, 1):
            beta = np.einsum('ij,ij',d, self.state['R'].get_subset((0, j+m), (2 * m + 1, 1)))

            r=self.state['R'].get((j, j + m))
            if r!=0:
                beta = beta /r

            d[j] = d[j] + (alpha[j]-beta)

        return d

    def _update_params(self,step_size,step_p):
        """
        Updates nn parameters with the step_p and step_size
        :param step_p: step tensor to add to the parameters
        :return:
        """
        with torch.no_grad():
            parameter_index = 0
            for p in self._params:
                if p.grad is not None:
                    param_shape = tuple(p.shape)
                    param_length=1
                    for i in p.shape:
                        param_length=param_length*i

                    p.add_(torch.as_tensor(step_p[parameter_index:parameter_index+param_length]).view(param_shape),
                           alpha=step_size)
                    parameter_index = parameter_index + param_length


    def _inner_product(self,vecs):
        """
        Calculates inner product of s_(k-1),y_(k-1),g_(k) with the passed vectors (vectors that are partitioned from b)
        :param vecs: dictionary containing the vectors for the inner products
        :return:
        """

        syg=vecs['syg']
        other_vecs=vecs['other']

        inner_product_results={'s':[],'y':[],'g':[]}
        for vec in other_vecs:

            if type(vec) == torch.Tensor:
                inner_product_results['s'].append(torch.dot(syg[0],vec).item())
                inner_product_results['y'].append(torch.dot(syg[1], vec).item())
                inner_product_results['g'].append(torch.dot(syg[2], vec).item())
            else:
                inner_product_results['s'].append(0.0)
                inner_product_results['y'].append(0.0)
                inner_product_results['g'].append(0.0)

        return inner_product_results

    def _inner_product_dicts_2_syg_all(self,inner_products):
        """
        Ensembles the inner product results to single lists for s_(k-1), y_(k-1), and g_(k)vectors
        :param inner_products:
        :return: all the inner products with order s_all, y_all, g_all
        """
        s_all=[]
        y_all=[]
        g_all=[]

        for inner_product in inner_products:
            s_all.extend(inner_product['s'])
            y_all.extend(inner_product['y'])
            g_all.extend(inner_product['g'])

        s_all = np.asarray(s_all).reshape(len(s_all), 1)
        y_all = np.asarray(y_all).reshape(len(y_all), 1)
        g_all = np.asarray(g_all).reshape(len(g_all), 1)

        return s_all,y_all,g_all

    def _add_tensors(self,coeffs_and_vecs):
        """
        Adds tensors by scaling with corresponding coefficients
        :param coeffs_and_vecs: dictionary containing coefficients and tensors
        :return:
        """
        coeffs=coeffs_and_vecs['coeffs']
        vecs=coeffs_and_vecs['vecs']

        tensor_shape=-1
        for i in range(len(vecs)):
            if vecs[i] != 'EMPTY':
                tensor_shape=tuple(vecs[i].shape)
                break

        if tensor_shape !=-1:
            superposed_tensor = torch.zeros(size=tensor_shape)

            for i in range(len(vecs)):
                if vecs[i] != 'EMPTY':
                    superposed_tensor.add_(vecs[i],alpha=coeffs[i])

            return superposed_tensor
        else:
            return torch.tensor(0,requires_grad=False)

    def _calculate_step_direction_p(self):

        # Calculate inner products s_all,y_all,g_all
        s_all = np.zeros(shape=(2 * self.defaults['m'] + 1, 1))
        y_all = np.zeros(shape=(2 * self.defaults['m'] + 1, 1))
        g_all = np.zeros(shape=(2 * self.defaults['m'] + 1, 1))

        if self.defaults['parallel_innerproduct'] == False:
            i = 0
            for vec in self.state['b'].q:
                if vec != 'EMPTY':
                    s_all[i] = torch.dot(vec, self.state['b'].get_last('s')).item()
                    y_all[i] = torch.dot(vec, self.state['b'].get_last('y')).item()
                    g_all[i] = torch.dot(vec, self.state['b'].get_last('g')).item()
                i = i + 1
        else:
            inner_product_results = \
                self.pool.map(self._inner_product, [{'syg': [self.state['b'].get_last('s'),
                                                             self.state['b'].get_last('y'),
                                                             self.state['b'].get_last('g')],
                                                     'other': self.state['b'].q[ \
                                                              self.partition_starts[i]: \
                                                              (self.partition_starts[i] + self.nof_products_per_cpu[
                                                                  i])]} \
                                                    for i in range(self.nof_cpu)])

            s_all, y_all, g_all = self._inner_product_dicts_2_syg_all(inner_product_results)

        # Update R
        self._update_R(s_all, y_all, g_all)

        # calculate d_(k): p_(k) = b_tensors_(k) * d_(k) from R
        d = self._calculate_d()

        # calculate step direction p_(k): x_(k+1)= x_(k) + step_size*p_(k)
        step_direction_p = torch.zeros_like(self.state['b'].get_last('g'), requires_grad=False)

        if self.defaults['parallel_superposition'] == False:
            i = 0
            for vec in self.state['b'].q:
                if vec != 'EMPTY':
                    step_direction_p.add_(vec, alpha=d[i, 0])
                i = i + 1

        else:
            tensor_addition_results = self.pool.map(self._add_tensors, [{'coeffs': d[ \
                                                                                   self.partition_starts[i]: \
                                                                                   (self.partition_starts[i] +
                                                                                    self.nof_products_per_cpu[i]), 0],
                                                                         'vecs': self.state['b'].q[ \
                                                                                 self.partition_starts[i]: \
                                                                                 (self.partition_starts[i] +
                                                                                  self.nof_products_per_cpu[i])]} \
                                                                        for i in range(self.nof_cpu)])
            for vec in tensor_addition_results:
                if type(vec) == torch.Tensor:
                    step_direction_p.add_(vec, alpha=1)

        return step_direction_p


    def _update_with_Wolfie(self, closure, loss_initial,step_direction_p):
        step_size = 1

        # Parameters for bisection method
        alpha = 0
        beta = np.Inf

        # Start line search with bisection
        line_search_iterations=0
        while line_search_iterations < self.defaults['line_search_max_iterations']:

            # Calculate gp inner product before stepping
            g_k_p_k = torch.dot(self.state['b'].get_last('g'), step_direction_p).item()

            # Step to a candidate location based on current step size
            self._update_params(step_size, step_direction_p)

            # Re-calculate loss and gradients
            Wloss = closure().item()

            # Calculate new gp inner product
            g_W=self._gather_flat_grad()
            g_W_p_k = torch.dot(g_W, step_direction_p).item()

            # Define (Weak) Wolfie Conditions
            def Wolfie1():
                return (Armijo_Rule(f_next=Wloss, f_initial=loss_initial, \
                                    c1=self.defaults['c1'], step_size=step_size, \
                                    pg_initial=g_k_p_k))

            def Wolfie2():
                return (Cuvature_Condition(pg_next=g_W_p_k, c2=self.defaults['c2'], pg_initial=g_k_p_k))

            # If the 1st Wolfie Condition is not satisfied decrease the step size with bisection method
            if (Wolfie1() == False):
                # return to the initial state
                self._update_params(-step_size, step_direction_p)

                # update bisection parameters and step size
                beta = step_size
                step_size = 0.5 * (alpha + beta)

            # If the 2nd Wolfie Condition is not satisfied increase the step size with bisection method
            elif (Wolfie2() == False):

                # if this is the last line search iteration and Wolfie1 is satisfied update b_g_last,b_y_last,b_s_last and break
                if line_search_iterations == self.defaults['line_search_max_iterations']-1:
                    self.state['b'].appnd('y', torch.add(g_W, self.state['b'].get_last('g'), alpha=-1))
                    self.state['b'].appnd('g', g_W)
                    self.state['b'].appnd('s', torch.mul(step_direction_p, other=step_size))

                    self.state['loss'] = Wloss
                    break

                # return to the initial state
                self._update_params(-step_size, step_direction_p)

                # update bisection parameters and step size
                alpha = step_size
                if beta == np.Inf:
                    step_size = 2 * alpha
                else:
                    step_size = 0.5 * (alpha + beta)


            # If all Wolfie Conditions are satisfied update b_g_last,b_y_last,b_s_last and break
            else:
                self.state['b'].appnd('y',torch.add( g_W, self.state['b'].get_last('g'), alpha=-1))
                self.state['b'].appnd('g', g_W)
                self.state['b'].appnd('s', torch.mul(step_direction_p, other=step_size))

                self.state['loss'] = Wloss
                break

            line_search_iterations=line_search_iterations+1

        # Check if an appropriate step size was found
        if (line_search_iterations > (self.defaults['line_search_max_iterations'])):
            # appropriate step size was not found, possible convergence to optimum
            return True
        else:
            return False

    def step(self, closure):
        """
        Optimization step
        :param closure: Re-evaluates cost and calculates gradients
        :return: loss for this step
        """

        closure = torch.enable_grad()(closure)
        self.state['loss'] = closure().item()

        # if s g and y are empty lists set the initial values
        if (self.state['b'].q[0]=='EMPTY'):
            # set initial g
            self.state['b'].appnd('g', self._gather_flat_grad())

            #set initial s and y as 0
            self.state['b'].appnd('s', torch.zeros_like(self._gather_flat_grad()))
            self.state['b'].appnd('y', torch.zeros_like(self._gather_flat_grad()))

        # start optimization
        while ((self.state['iteration'] < self.defaults['max_iteration']) and \
                (self.state['loss'] >self.defaults['termination_threshold'])):

            # calculate the step direction p
            step_direction_p = self._calculate_step_direction_p()

            # if Wolfie conditions are required to be satisfied update accordingly by setting step_size with line search
            if self.defaults['check_Wolfie']==True:

                possible_convergence = self._update_with_Wolfie(closure = closure,\
                                         loss_initial=self.state['loss'],step_direction_p=step_direction_p)

                # stop optimization step if the update with wolfie conditions were unsuccesful
                if possible_convergence:
                    break

            # if Wolfie conditions are not required to be satisfied update with a fixed step_size
            else:
                step_size = self.defaults['step_size']
                self._update_params(step_size, step_direction_p)

                # Re-calculate loss and gradients
                self.state['loss']=closure().item()

                # update b_g_last,b_y_last,b_s_last
                g_new=self._gather_flat_grad()

                self.state['b'].appnd('y', torch.add(g_new, self.state['b'].get_last('g'), alpha=-1))
                self.state['b'].appnd('g', g_new)
                self.state['b'].appnd('s', torch.mul(step_direction_p, other=step_size))

            self.state['iteration'] = self.state['iteration'] + 1

        # reset number of iterations and loss
        loss=self.state['loss']
        self.reset_states(reset_loss=True,reset_iteration=True)

        # return the loss calculated for this step
        return loss

    def reset_states(self,reset_b=False,reset_R=False,reset_iteration=False,reset_loss=False):
        """
            Resets desired states.
        :param reset_b:
        :param reset_R:
        :param reset_iteration:
        :param reset_loss:
        :return:
        """
        m=self.defaults['m']
        if(reset_R):
            self.state['R'] = SymmetricMatrix(2 * m + 1)
        if(reset_b):
            self.state['b'] = MultiQueue({'s': m,
                                          'y': m,
                                          'g': 1})
        if(reset_loss):
            self.state['loss']=np.Inf
        if(reset_iteration):
            self.state['iteration'] = 0
