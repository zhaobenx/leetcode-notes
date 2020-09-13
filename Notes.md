# Leetcode notes

## Highlight

### [14](#14). Python built-in sort

### 22. [Catalan number](https://zh.wikipedia.org/wiki/卡塔兰数)

$C_n = \frac{1}{n+1}{2n \choose n} = \frac{(2n)!}{(n+1)!n!}$ $C_n$表示有*2n+1*个节点组成不同构满[二叉树](https://zh.wikipedia.org/wiki/二叉树)（full binary tree）的方案数

Also true for 96

### 23. PriorityQueue 用来排序，put(),get()

### 35. `itertools.groupby()`

### 39. 递归时尽量减少层数，尽早筛除无用分支

### 53. 动态规划

### 59. ` zip(*A[::-1])` rotate A clockwise.

### 74. `bisect` for binary search.

### 875.  -(-a//b) means ceil(a/b)

## ## TODO:

- [ ] 5
- [ ] 16
- [ ] 18 4 sum
- [ ] 33 
- [ ] 60 permutation 
- [ ] 73 2d binary search
- [ ] 146 lru cache
- [ ] 216
- [ ] 329 with dp

---

## 3. [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

### 动态规划，合理利用字典

> Given a string, find the length of the **longest substring** without repeating characters.

### Idea

1. Brute force(Too slow)
2. Dynamic programing. Set a list `start`, to store the local longest substring, so if one substring is from i to j, `start[j] = i`. Loop over the string, detect if the new letter in the last substring, if so, create a new substring for this letter, else add the letter to the substring.
3. Count each letter’s index, find the max interval. And find the minimum of all the interval.:x:(cannot give answer like “aab”)
4. Another approach is to store each letters’ last position into a `dict`, and during the loop, just compare local length to the last letter position.

## 6. [ZigZag Conversion](https://leetcode.com/problems/zigzag-conversion/)

> The string `"PAYPALISHIRING"` is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)

### Idea

1. Find a formula to generate the sequence, and combine it to a str. 

## 7. [Reverse Integer]( https://leetcode.com/problems/reverse-integer/) <a name="7"></a>

> Given a 32-bit signed integer, reverse digits of an integer.

### Idea

1. Use python `str` and `int` to convert number to string and back.
2. Convert digit by digit.（Same speed with method 1)

### Code

```python
class Solution:
    def reverse(self, x: int) -> int:
        sign = 1 if x>0 else -1
        res = int(str(abs(x))[::-1])*sign
        return res if -2147483648< res <2147483647 else 0
```

## 9. [Palindrome Number](https://leetcode.com/problems/palindrome-number/)

> Determine whether an integer is a palindrome. An integer is a palindrome when it reads the same backward as forward.

### Idea

1. Same as [7](#7)

## 11. [Container With Most Water](https://leetcode.com/problems/container-with-most-water/)

### 分治

> Given *n* non-negative integers *a1*, *a2*, ..., *an* , where each represents a point at coordinate (*i*, *ai*). *n* vertical lines are drawn such that the two endpoints of line *i* is at (*i*, *ai*) and (*i*, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water.

### Idea

1. BF(time exceeded) :x:
2. Two Pointer Approach

## 12. [Integer to Roman](https://leetcode.com/problems/integer-to-roman/)

> Roman numerals are represented by seven different symbols: `I`, `V`, `X`, `L`, `C`, `D` and `M`.

### Idea

1. Store the value from I to M, then for loop to add the value to result.
2. Calculate all possible value e.g CM,M,MM,MMM… Then this problem can be solved in O(N).:heavy_check_mark:

## 13. [Roman to Integer](https://leetcode.com/problems/roman-to-integer/)

### 注意循环

### Idea

1. If one Roman is less than its latter one, minus it, otherwise add it. :exclamation:Carefully handle the loop about where to start and where to end.

## 14. :star:[Longest Common Prefix](https://leetcode.com/problems/longest-common-prefix/) <a name="14"></a>

### 脑子要活

> Write a function to find the longest common prefix string amongst an array of strings

### Idea

1. Vertical scanning
2. Sort the string list first, then compare the first string and the last string. :star:
3. Use python zip, to vertically split strings and use set to compare.
4. First get the shortest string, then compare to other strings.

### Snippet

`startwith`, `zip`, `set`

## 15. [3Sum](https://leetcode.com/problems/3sum/)

### 代码严谨

> Given an array `nums` of *n* integers, are there elements *a*, *b*, *c* in `nums` such that *a* + *b* + *c* = 0? Find all unique triplets in the array which gives the sum of zero.

### Idea

1. Same as 2Sum, first calculate two pairs sum and store them in a dict. This method takes O(N^2^), which exceeds the time limits. :x:
2. Also O(N^2^). First sort the array, and the first loop from start, the second use two pointer approach from end and beginning.  Need to carefully handle the condition in different loops.

## 16. 3Sum Closest

Same as 15

## 17. [Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)

### 递归

> Given a string containing digits from `2-9` inclusive, return all possible letter combinations that the number could represent.

### Idea

1. Recursion

### Code

```python
def h(now, lists):
    if not lists:
        return [now,]
    res = []
    for i in lists[0]:
        res.extend(h(now + i, lists[1:]))
    return res
```

## 19. [Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)

> Given a linked list, remove the *n*-th node from the end of list and return its head.

### Idea

1. Simple iteration, need to take care of how to delete the first value. But this is two pass.
2. No really one pass. Just uses two pointers, one follows the other behind n nodes. 

## 20. [Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)

### 栈

> Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

### Idea

1. Set three counters count if the counter is always no less than zero. :x:*“({)}”*

2. Stack

### Snippet 

`dict = {")": "(", "}": "{", "]": "["}`   

## 21. [Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)

> Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists.

### Idea

1. Pointer loop. :exclamation: This problem request the smaller one comes at the first. e.g. [2] [1] → [1, 2]

### Snippet

List-wise `and` and `or` . 

```python
while l1 and l2:
    if l1.val < l2.val:
        ptr.next = l1
        l1 = l1.next                
    else:
        ptr.next = l2
        l2 = l2.next                
        ptr = ptr.next

ptr.next = l1 or l2
```



## 22. :star:[Generate Parentheses](https://leetcode.com/problems/generate-parentheses/)

### 递归、DP

> Given *n* pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

### Idea

1. Make this problem recursive, r(n) = ()+r(n-1) ∪(+r(n-1) +) ∪ r(n-1)+(). :x: If n == 4, this cannot explain `(())(()) `
2. Change the recursive to a DP problem, r(n) = (+r(n-1)+) ∪ $\cup_{i=1}^{n-1}r(i)+r(n-i)$.:heavy_check_mark: or r(n) = $\cup_{i=0}^{n-1}(+r(i)+)+r(n-i)$
3. Recursive, but to find a place to insert one pair of parentheses. 
4. Divide the problems into subproblems: to add `(` or to close brackets. :heavy_check_mark:

### Note

:exclamation: For python `sets = sets.union(...)` to update a `set`.

## 23.:star: [Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)

### 优先队列

> Merge *k* sorted linked lists and return it as one sorted list. Analyze and describe its complexity.

### Idea

1. Simply merge. (time exceed:x:)
2. Divide and conquer. Separate the lists to two parts, and if one separated contains only two, merge this two like [#21](21).
3. :sweat: Convert node to list, then sort, then convert back.
4. Python built-in `PriorityQueue` :star:

## 24. [Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/)

> Given a linked list, swap every two adjacent nodes and return its head.

### Idea

1. Basic pointer operation. Should draw a flow chart before type a code.

## 26. [Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)
### 循环注意开始和结束以及状态量的初始值

> Given a sorted array *nums*, remove the duplicates [**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm) such that each element appear only *once* and return the new length.
>
> Do not allocate extra space for another array, you must do this by **modifying the input array in-place** with O(1) extra memory.

### Idea

1. Pointer. Conquer and divide should take care of the initial value.  
2. Use python `set`

### Note

Return the length of the returned array and change the array in-place.

### Code

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        length = 1        
        for i in range(1, len(nums)):
            if nums[i] != nums[i - 1]:
                nums[length] = nums[i]
                length += 1
        return length             
```

## 27. [Remove Element](https://leetcode.com/problems/remove-element/)

> Given an array *nums* and a value *val*, remove all instances of that value [**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm) and return the new length.

### Idea

1. Same as [26]().

## 28. [Implement strStr()](https://leetcode.com/problems/implement-strstr/)

> Implement [strStr()](http://www.cplusplus.com/reference/cstring/strstr/).

### Idea

1. For loop in haystack, and compare with needle. :exclamation: For loop range is from 0 to `len(H) - len(N) + 1`.
2. Python built-in `str.find()` ​.​ ​X​D:stuck_out_tongue_closed_eyes:

## 35. [Search Insert Position](https://leetcode.com/problems/search-insert-position/)

### 二分查找

> Given a sorted array and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

### Idea

1. Simple traversal.
2. Binary search.

### Snippet

`mid = start + (end - start)//2`, `end = mid`, `start = mid + 1`

## 38. [Count and Say](https://leetcode.com/problems/count-and-say/)

### Idea

1. Recursively read last string.  But this is too slow
2. Python built-in `itertools.groupby()`, return a list of `[(element, iter), …]` is a good choice.
3. Prepare the results in advance.

## 39. [Combination Sum](https://leetcode.com/problems/combination-sum/)

### 减少递归层数

> Given a **set** of candidate numbers (`candidates`) **(without duplicates)** and a target number (`target`), find all unique combinations in `candidates` where the candidate numbers sums to `target`.

### Idea 

1. Simple recursive. No matter with or without DP, the speed are slow. **One way** is instead of using set() to avoid duplicate, just add a `index` parameter, so the result is always in ascending order.
2. :exclamation: Try to avoid even just one more function call e.g.

```python 
# bad
def dfs(i):
    if i > xxx:
        return
    for j in range(i)
    	dfs(j-1)
# good  
def dfs(i):
    for j in range(i):
        if j > xxx:
            break
        dfs(j-1)   
    
```

## 40. [Combination Sum II](https://leetcode.com/problems/combination-sum-ii/)

> Given a collection of candidate numbers (`candidates`) and a target number (`target`), find all unique combinations in `candidates` where the candidate numbers sums to `target`.

### Idea

1. Same as 39, just modify the loop index to avoid duplication. Also need to check if elements are the same.

## 46. [Permutations](https://leetcode.com/problems/permutations/)

> Given a collection of **distinct** integers, return all possible permutations.

### Idea

1. Recursion
2. Python built-in `itertools.permutations()`

## 47. [Permutations II](https://leetcode.com/problems/permutations-ii/)

> Given a collection of numbers that might contain duplicates, return all possible unique permutations.

### Idea

1. Same as 46
2. Another solution is to insert, code shown below(:exclamation:this code ignore duplation handling).

### Code

```python
def permute(nums):
    permutations = [[]]    
    for head in nums:
        permutations = [rest[:i]+[head]+rest[i:] for rest in permutations for i in range(len(rest)+1)]        
    return permutations
```

## 48. [Rotate Image](https://leetcode.com/problems/rotate-image/)

> You are given an *n* x *n* 2D matrix representing an image.
>
> Rotate the image by 90 degrees (clockwise).

### Idea

1. Simple for loop, for each element in i,j exchange four element that with rotational symmetry. (Note to handle if matrix length is odd).
2. Amazing python code `matrix[::] = reversed(zip(*matrix))`

## 49. [Group Anagrams](https://leetcode.com/problems/group-anagrams/)

> Given an array of strings, group anagrams together.

### Idea

1. Use sorted words as the key, word as the value. 
2. Use a 26-length tuple as the key.
3. Assign different primes to letter, then the sum of each word as the key.

## 50. [Pow(x, n)](https://leetcode.com/problems/powx-n/)

> Implement [pow(*x*, *n*)](http://www.cplusplus.com/reference/valarray/pow/), which calculates *x* raised to the power *n* (x^n^).

### Idea

1. Python built-in `math.pow()`
2. Normal iteration :x:(too slow)
3. Binary search e.g. pow(x,25)->pow(x,12)*pow(x,13) (with DP).

## 53.:star: [Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)

### 动态规划

> Given an integer array `nums`, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

### Idea

1. Divide and conquer. If use code in CLRS for the MAXIMUM-SUBARRAY, the time complexity will be `O(nlog n)`, which is not as excepted. 
2. Just simple DP. This is a simple problems, as you don’t need to record the maximum subarray, you just need to remember the largest result. :star:(Kadane's Algorithm)

### Code

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        memo = [nums[0]] * len(nums)
        for i in range(1, len(nums)):
            memo[i] = max(memo[i - 1] + nums[i], nums[i])
        return max(memo)
```

## 58. Length of Last Word

Easy, python built-in `strip()`,`split()`,`len()`. 

## 69. [Sqrt(x)](https://leetcode.com/problems/sqrtx/)

### 二分、牛顿

### Idea

1. `math.sqrt()`
2. Binary search
3. Newton‘s method

## 169. [Majority Element](https://leetcode.com/problems/majority-element/)

### BM算法

> Given an array of size *n*, find the majority element. The majority element is the element that appears **more than** `⌊ n/2 ⌋` times.

### Idea

1. Set one variables as the dominant element `m` and one counter `t`. In iteration, if `i` is `m` then add 1 to counter, else minus 1. If the counter is less than 0, set `i` as new dominant element and **set counter** to 0. (***Boyce-Moore Algorithms***)
2. Use python built-in `collection.Counter`. (not as fast as method 1)

### Code

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        m = nums[0]
        t = 0
        for i in nums[1:]:
            if i == m:
                t += 1
            else:
                t -= 1
                
            if t<0:
                m = i
                t = 0
        return m
```

## 216. [Combination Sum III](https://leetcode.com/problems/combination-sum-iii/)

> Find all possible combinations of ***k*** numbers that add up to a number ***n***, given that only numbers from 1 to 9 can be used and each combination should be a unique set of numbers.

### Idea

1. BF(Too slow)
2. Optimize the recursion by checking some conditions.

## 263. Ugly Number](https://leetcode.com/problems/ugly-number)

> Write a program to check whether a given number is an ugly number.
>
> Ugly numbers are **positive numbers** whose prime factors only include `2, 3, 5`.

Simple.  30**32%num

## 336. :star:[Palindrome Pairs](https://leetcode.com/problems/palindrome-pairs/)

### 巧妙的循环

> Given a list of **unique** words, find all pairs of **distinct** indices `(i, j)` in the given list, so that the concatenation of the two words, i.e. `words[i] + words[j]` is a palindrome.

### Idea

1. BF :x:(time exceeded)
2. Add the last letter to a dict, then iteration based on those dict. ( still too slow)
3. :star:First add reversed word into a dictionary. Then for each word, split it in to prefix and suffix (:exclamation: for loop with range `n + 1` , only in this way prefix and suffix can cover all the word) .  Then check if the prefix in the reversed dictionary and if the suffix is self palindromic. And vice versa.  (:exclamation: Avoid check the whole words twice.)

```
          IN?
    +-----------------+
    |                 |
    |                 v
+---+---+-------+  +--+--+
|prefix |suffix |  |table|
+-------+---+---+  +-----+
            |
            v
    +-------+-------+
    |IS palindromic?|
    +---------------+
```

### Code

```python
class Solution:
    def palindromePairs(self, words: List[str]) -> List[List[int]]:
        # assume all letters is in lower case
        table = {word[::-1]:i for i, word in enumerate(words)}
        
        def is_p(word):
            return True if word == word[::-1] else False
        
        res = []
        for index, word in enumerate(words):
            for j in range(len(word) + 1):
                pre = word[:j]
                suf = word[j:]
                if  pre in table and is_p(suf):
                    if table[pre] != index:
                        res.append([index, table[pre]])
                if j > 0 and suf in table and is_p(pre):
                    if table[suf] != index:
                        res.append([table[suf], index])                
    
        return res
```



## 509. [Fibonacci Number](https://leetcode.com/problems/fibonacci-number/)

### Idea

1. Iteration
2. Recursion

## 31. [Next Permutation](https://leetcode.com/problems/next-permutation/)

### 记住

> Implement **next permutation**, which rearranges numbers into the lexicographically next greater permutation of numbers.

1. First find the last ascending element *i*. Then sort the list behind *i*. Then find the minimal element that greater than *i*, and swap this with *i*. (Cannot explain why, don’t ask me.)
2. **Official answer**. First find the last ascending element *i*. **Then find the one element that greater than *i*** , and swap this with *i*.  Then reverse the list behind *i*. ![gif](https://leetcode.com/media/original_images/31_Next_Permutation.gif)



## 33. [Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)

### 还需再想

> Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

### Idea 

1. bad idea. First use binary search find the ‘mid’ points, which is the first one less than the first element. Then find the target element in terms of the ‘mid’ point. $O(log{N})$
2. Use just one while loop, split the problem into four condition: ~~mid is less than first, target on mid right; mid is less than first, target on mid right;~~ **TODO**

## 34. [Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

> Given an array of integers `nums` sorted in ascending order, find the starting and ending position of a given `target` value.

### Idea

1. Too slow, two binary search, find the minimal target and the maximum target.
2. :bulb: Add a judge after the first binary search. To check if the element exists.

## 36. [Valid Sudoku](https://leetcode.com/problems/valid-sudoku/)

> Determine if a 9x9 Sudoku board is valid.

### Idea

1. Loop over the board, check if there is any duplication in rows, columns, grids.

## 54. [Spiral Matrix](https://leetcode.com/problems/spiral-matrix/)

> Given a matrix of *m* x *n* elements (*m* rows, *n* columns), return all elements of the matrix in spiral order.

### Idea

1. Loop, find the rules: 1 main loop from 0 to half of the minimum of sizes; 4 sub loop loop from 4 edges.(Note to handle one row/column loop back problems in the center row/column)

## 55. :star:[Jump Game](https://leetcode.com/problems/jump-game/)

> Given an array of non-negative integers, you are initially positioned at the first index of the array.

### Idea

1. DFS (failed, time exceeds and stack overflow e.g. 100000,999999,9999998,….,1,0,0). But if using stack instead of using recursion, the speed can be a little bit faster(only beat 5%).
2. Greedy. Track from back to front. To trace the last position one point can reach.:star:

### Code

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        last_postion = len(nums) - 1
        for i in range(len(nums) -1, -1, -1):
            if nums[i] + i >= last_postion:
                last_postion = i

        return last_postion == 0 
```



## 66. [Plus One](https://leetcode.com/problems/plus-one/)

> Given a **non-empty** array of digits representing a non-negative integer, plus one to the integer.

### Idea

1. Python built-in map, str, list comprehension. 
2. For loop.

## 56. [Merge Intervals](https://leetcode.com/problems/merge-intervals/)

> Given a collection of intervals, merge all overlapping intervals.

### Idea

1. Greedy algorithm, sort the array first, then loop over the array. O(nlogn )

## 59. [Spiral Matrix II](https://leetcode.com/problems/spiral-matrix-ii/)

> Given a positive integer *n*, generate a square matrix filled with elements from 1 to *n*2 in spiral order.

### Idea

1. Like [54](#54), use for loop to generate index.
2. Amazing idea from leetcode.

    ```python
    def generateMatrix(self, n):
        A, lo = [], n*n+1
        while lo > 1:
            lo, hi = lo - len(A), lo
            A = [range(lo, hi)] + zip(*A[::-1])
        return A
    ```

## 89. [Gray Code](https://leetcode.com/problems/gray-code/)

> The gray code is a binary numeral system where two successive values differ in only one bit.

### Idea

1. :star:	Amazing gray(n) = n^(n>>1)

## 61. [Rotate List](https://leetcode.com/problems/rotate-list/)

> Given a linked list, rotate the list to the right by *k* places, where *k* is non-negative.

### Idea 

1. Simple loop. First count the length, then calculate the position where to concatenate. 

## 62. [Unique Paths](https://leetcode.com/problems/unique-paths/)

> A robot is located at the top-left corner of a *m* x *n* grid (marked 'Start' in the diagram below).

### Idea

1. Use combination???? Still thinkin why. I see. There are totally m + n - 2 steps, which include m - 1 step to go right or n - 1 to go down. So the total paths is ${n + m -2 \choose m - 1}$ or $n + m - 2 \choose n - 1$. :heavy_check_mark:
2. Find a rules, that it is a sum of sum of sum of …. of the range to length, which the number of sum is the width -1.

## 71. [Simplify Path]( https://leetcode.com/problems/simplify-path/ ) 

>  Given an **absolute path** for a file (Unix-style), simplify it. Or in other words, convert it to the **canonical path**. 

### Idea

1. Simple stack.

##  73. [Set Matrix Zeroes ]( https://leetcode.com/problems/set-matrix-zeroes/ )

>  Given a *m* x *n* matrix, if an element is 0, set its entire row and column to 0. Do it [**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm). 

### Idea

1. Space complexity is $O(m + n)$. Loop over matrix, record all the rows and columns that contains  0, then set them to zero.
2. Space complexity is constant, but time complexity will be $O(nm\times(m+n))$. Just loop over, and set every rows and columns to None. And then replace None with 0.
3. :heavy_check_mark: Space constant, time complexity is $O(mn)$. Set the first row and column as the mark of each row or column as zero or not.

##  75. [Sort Colors ]( https://leetcode.com/problems/sort-colors/ )

>  Given an array with *n* objects colored red, white or blue, sort them **[in-place](https://en.wikipedia.org/wiki/In-place_algorithm)** so that objects of the same color are adjacent, with the colors in the order red, white and blue. 

### Idea

1. Two pass with counting sort.
2. :heavy_check_mark: One pass with three pointer. Like what is done in quicksort’s partition. Two identity red and white’s right border, one identity blue’s left border. (Note, the termination condition is white > blue) .

##  77. [Combinations](https://leetcode.com/problems/combinations/)

>  Given two integers *n* and *k*, return all possible combinations of *k* numbers out of 1 ... *n*. 

### Idea

1. Use recursion.

2. A non-recursive method. Maintain two lists, one for result, one for middle variables; Use  loop from 1 to n,   the try to append this loop variables to the end of each middle lists, if the length reached to k, save the list to result.

   Note to optimize useless branches. 

### Code

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []
        mid = []
        for i in range(1, n + 2):
            tmp = []
            for p in mid:
                if len(p) == k:
                    res.append(p)
                elif len(p) >= k - n + i - 1:
                    tmp.append(p + [i])
                    tmp.append(p)
            tmp.append([i])
            mid = tmp
        return res
```



##  74. [Search a 2D Matrix ]( https://leetcode.com/problems/search-a-2d-matrix/)

### 读题！

> Write an efficient algorithm that searches for a value in an *m* x *n* matrix. This matrix has the following properties:
>
> - Integers in each row are sorted from left to right.
> - The first integer of each row is greater than the last integer of the previous row.

### Idea 

1. Treat the matrix as a sorted array, do binary search on it. 
2. :heavy_check_mark:Python built-in `bisect` is much faster than hand-written code.

##  78. [Subsets]( https://leetcode.com/problems/subsets/ ) 

>  Given a set of **distinct** integers, *nums*, return all possible subsets (the power set). 

### Idea

1. There are $2^n$ subsets, so to convert one number(from 0 to $2^n -1$) to binary array. Using the binary to select each elements.
2. Append one element to result each time. 

### Code 

```python 
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = [[]]
        for i in nums:
            res += [j+[i] for j in res]
        return res
```

Much faster:

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = [[]]
        for n in nums:
            for i in range(len(res)):
                res.append(res[i] + [n])
        return res
```

Comparison: Append is the fastest, then is expand, then is +=.



## 64. [Minimum Path Sum](https://leetcode.com/problems/minimum-path-sum/)

### 用循环

> Given a *m* x *n* grid filled with non-negative numbers, find a path from top left to bottom right which *minimizes* the sum of all numbers along its path.

### Idea

1. Simple dp, loop search for all solutions. :x: Too slow.
2. Loop in reverse order, calculate the minimum cost. Still too slow.
3. Use Loop in DP is much faster than recursive. :heavy_check_mark:

## 70. [Climbing Stairs](https://leetcode.com/problems/climbing-stairs/)

> You are climbing a stair case. It takes *n* steps to reach to the top.
>
> Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

### Idea

1. Fibonacci sequence.

## 94. [Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)

> Given a binary tree, return the *inorder* traversal of its nodes' values.

### Idea

1. Recursion.
2. Iteration. Key point is to transform recursion to iteration. :star:

## 96. [Unique Binary Search Trees](https://leetcode.com/problems/unique-binary-search-trees/)

> Given *n*, how many structurally unique **BST's** (binary search trees) that store values 1 ... *n*?

### Idea

1. Note that one BST can only have one sequence of 1…n, thus we can just count  the possible shape of BST. That is to count all the possible left and right children combinations.  Use lru to make this a DP.
2. Catalan number

## 101. [Symmetric Tree](https://leetcode.com/problems/symmetric-tree/)

> Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).

### Idea

1. Two pointer, do a DFS search.

## 102. [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)

> Given a binary tree, return the *level order* traversal of its nodes' values. (ie, from left to right, level by level).

### Idea

1. BFS, with two counter. Counter counts next level node numbers, one indicates now level numbers.

## 104. [Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)

> Given a binary tree, find its maximum depth.

### Idea 

1. DFS with level on each walk.



## 139. [Word Break](https://leetcode.com/problems/word-break/)

> Given a **non-empty** string *s* and a dictionary *wordDict* containing a list of **non-empty** words,

### Idea

1. DP, like longest subarray.



## 200. [Number of Islands](https://leetcode.com/problems/number-of-islands/)

> Given a 2d grid map of `'1'`s (land) and `'0'`s (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

### Idea 

1. Iterate over each grid, and  flip all the adjacent 1 to 0. Count the number of first flip. 



## 141. [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)

> Given a linked list, determine if it has a cycle in it.

### Idea

1. Obtain a set to store all the visited node. Time is O(n), space is O(n).
2. Two pointer, one is slow, one is fast(normal). If fast one catch the slower one, there is a cycle. Time O(n), Space O(1).

## 142. [Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/)

> Given a linked list, return the node where the cycle begins. If there is no cycle, return `null`.

### Idea 

1. Same as [141](141. [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/))

## 329. [Longest Increasing Path in a Matrix](https://leetcode.com/problems/longest-increasing-path-in-a-matrix/description/)

> Given an integer matrix, find the length of the longest increasing path.

### Idea

1. Simple traverse over the whole matrix, and find the maximum. :x: Too slow.
2. DP

## 875. [Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/)

> Return the minimum integer `K` such that she can eat all the bananas within `H` hours.

### Idea

1. Binary search, search for the minimum K within 1 to max(piles).



## 92. [Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/)

> Reverse a linked list from position *m* to *n*. Do it in one-pass.

### Idea

1. Carefully reverse. Draw a plot before coding.

## 207. [Course Schedule](https://leetcode.com/problems/course-schedule/)

> Given the total number of courses and a list of prerequisite **pairs**, is it possible for you to finish all courses?

### Idea

1. DFS, to detect if there is cycle

#### Note

**Carefully use defaultdict** as there my be unexcepted result. e.g. `if some in default_dict[not_existed_key]`


---

## Conclusion

1. For python, `if` is faster than `max`, `min`. Although it is not pythonic. 