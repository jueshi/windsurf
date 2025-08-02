# Remove duplicate numbers
nums=[1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9]
unique_numbers = list({num for num in nums if num != nums[0]})
print(unique_numbers)