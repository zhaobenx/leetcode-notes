# Leetcode notes

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

## 14. :star:[Longest Common Prefix](https://leetcode.com/problems/longest-common-prefix/)

### 脑子要活

> Write a function to find the longest common prefix string amongst an array of strings

### Idea

1. Vertical scanning
2. Sort the string list first, then compare the first string and the last string.
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

## Conclusion

1. For python, `if` is faster than `max`, `min`. Although it is not pythonic. 