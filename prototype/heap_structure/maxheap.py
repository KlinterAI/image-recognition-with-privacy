# First let us complete a minheap data structure.
# Please complete missing parts below.
class MaxHeap:
    def __init__(self):
        self.H = [None]
        
    def size(self):
        return len(self.H)-1
    
    def __repr__(self):
        return str(self.H[1:])
        
    def satisfies_assertions(self):
        for i in range(2, len(self.H)):
            assert self.H[i] <= self.H[i//2],  f'Maxheap property fails at position {i//2}, parent elt: {self.H[i//2]}, child elt: {self.H[i]}'
    
    def max_element(self):
        return self.H[1]

    ## bubble_up function at index
    ## WARNING: this function has been cut and paste for the next problem as well 
    def bubble_up(self, index):
        # your code here
        assert index >= 1
        if index == 1: 
            return 
        parent_index = index // 2
        if self.H[parent_index] > self.H[index]:
            return 
        else:
            self.H[parent_index], self.H[index] = self.H[index], self.H[parent_index]
            self.bubble_up(parent_index)

    ## bubble_down function at index
    ## WARNING: this function has been cut and paste for the next problem as well 
    def bubble_down(self, index):
        # your code here
        assert index >= 1 and index < len(self.H)
        lchild_index = 2 * index
        rchild_index = 2 * index + 1
        # set up the value of left child to the element at that index if valid, or else make it +Infinity
        lchild_value = self.H[lchild_index] if lchild_index < len(self.H) else float('-inf')
        # set up the value of right child to the element at that index if valid, or else make it +Infinity
        rchild_value = self.H[rchild_index] if rchild_index < len(self.H) else float('-inf')
        # If the value at the index is lessthan or equal to the minimum of two children, then nothing else to do
        if self.H[index] >= max(lchild_value, rchild_value):
            return 
        # Otherwise, find the index and value of the smaller of the two children.
        # A useful python trick is to compare 
        max_child_value, max_child_index = max ((lchild_value, lchild_index), (rchild_value, rchild_index))
        # Swap the current index with the least of its two children
        self.H[index], self.H[max_child_index] = self.H[max_child_index], self.H[index]
        # Bubble down on the minimum child index
        self.bubble_down(max_child_index)

    # Function: insert
    # Insert elt into minheap
    # Use bubble_up/bubble_down function
    def insert(self, elt):
        # your code here
        index = len(self.H)
        self.H.append(elt)
        if index == 1:
            return
        parent_index = index // 2
        parent_index_2 = (index-1) // 2
        if(parent_index == parent_index_2 and self.H[parent_index] < min(self.H[index], self.H[index-1])):
            self.bubble_down(parent_index)
        elif self.H[parent_index] < self.H[index]:
            self.bubble_up(index)
        else:
            return

    # Function: heap_delete_min
    # delete the smallest element in the heap. Use bubble_up/bubble_down
    # Function: delete_max
    # delete the largest element in the heap. Use bubble_up/bubble_down
    def delete_max(self):
        # your code here
        last_index = self.size()
        self.H[1], self.H[last_index] = self.H[last_index], self.H[1]
        self.H.pop(last_index)
        if self.size() > 1:
            self.bubble_down(1)

    