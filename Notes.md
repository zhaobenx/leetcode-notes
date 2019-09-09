# Leetcode notes

## Highlight

### [14](#14). Python built-in sort

### 22. [Catalan number](https://zh.wikipedia.org/wiki/卡塔兰数)

$C_n = \frac{1}{n+1}{2n \choose n} = \frac{(2n)!}{(n+1)!n!}$ $C_n$表示有*2n+1*个节点组成不同构满[二叉树](https://zh.wikipedia.org/wiki/二叉树)（full binary tree）的方案数

### 35. `itertools.groupby()`



---

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



## Conclusion

1. For python, `if` is faster than `max`, `min`. Although it is not pythonic. 