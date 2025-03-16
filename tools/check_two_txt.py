from collections import Counter

def compare_text_files(file1, file2):
    # Read the files
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        text1 = f1.read()
        text2 = f2.read()
    
    # Adjust the length of the texts
    max_length = max(len(text1), len(text2))
    text1 = text1.ljust(max_length)
    text2 = text2.ljust(max_length)
    
    # Find differences
    different_chars = [char1 for char1, char2 in zip(text1, text2) if char1 != char2]
    
    # Count the number of different characters
    num_differences = len(different_chars)
    
    # Count the most frequent different characters
    char_counts = Counter(different_chars)
    
    print(f"Number of different characters: {num_differences}")
    #print("All different characters and their frequency:")
    #for char, count in char_counts.items():
    #    print(f"'{char}': {count}")

    
for i in range(0,5):
    print(f"trial {i} compared to zeroes")
    compare_text_files(f'/Coding/CVPR2025_abaw_framewise/data_abaw/test_data/output/trial-{i}.txt', '/Coding/CVPR2025_abaw_framewise/data_abaw/test_data/submissions (examples)/trial-0.txt')


for i in range(0,5):
    for j in range(0,5):
        print(f"trial {i} compared to trial {j}")
        compare_text_files(f'/Coding/CVPR2025_abaw_framewise/data_abaw/test_data/output/trial-{i}.txt', f'/Coding/CVPR2025_abaw_framewise/data_abaw/test_data/output/trial-{j}.txt')