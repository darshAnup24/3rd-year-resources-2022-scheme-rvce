import matplotlib.pyplot as plt
import random
import re


# Minimal Stopwords (customizable)

STOPWORDS = {
    'the','is','in','and','to','of','a','for','on','with','that','this','it',
    'as','at','by','an','be','are','was','were','from','or','but','not',
    'have','has','had','you','i','we','they','he','she','them','his','her',
    'my','your','our','their'
}


# Preprocessing Function

def preprocess(text):
    # Lowercase
    text = text.lower()
    
    # Extract words (alphanumeric)
    words = re.findall(r'\b[a-z0-9]+\b', text)

    # Count frequencies excluding stopwords
    freq_dict = {}
    for word in words:
        if word not in STOPWORDS:
            freq_dict[word] = freq_dict.get(word, 0) + 1

    return freq_dict


# Word Cloud Visualization

def wordcloud(freq):
    plt.figure()

    #sort the dictionary
    freq=sorted(freq.items(),key=lambda x:x[1], reverse=True)

    grid_size=10
    grid=[]

    for i in range(grid_size):
        row=[False]*10
        grid.append(row)

    for key, val in freq:
        x, y = random.randint(0, grid_size-1), random.randint(0, grid_size-1)

        while grid[x][y]:
            x, y = random.randint(
                0, grid_size-1), random.randint(0, grid_size-1)
        grid[x][y]=True

        color = (random.random(), random.random(), random.random())
        plt.text(x/grid_size,y/grid_size,key,fontsize=val*10, color=color)
    plt.show()


# Main Execution


text=""" Born in Ranchi, Dhoni made his first class debut for Bihar in 1999. He made his debut for the Indian cricket team on 23 December 2004 in an ODI against Bangladesh and played his first test a year later against Sri Lanka. In 2007, he became the captain of the ODI side before taking over in all formats by 2008. Dhoni retired from test cricket in 2014, but continued playing in limited overs cricket till 2019. He has scored 17,266 runs in international cricket including 10,000 plus runs at an average of more than 50 in ODIs. """

freq = preprocess(text)

print("Word Frequencies:")
for word, count in sorted(freq.items(), key=lambda x: x[1], reverse=True):
    print(f"{word}: {count}")

wordcloud(freq)