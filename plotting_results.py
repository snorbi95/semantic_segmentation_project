import collections
import os

import matplotlib.pyplot as plt


def plot_distribution():
    import matplotlib.pyplot as plt

    artery_dir = f'images'
    artery_images = os.listdir(artery_dir)

    # ureter_dir = f'ureter_dataset/ureter_images'
    # ureter_images = os.listdir(ureter_dir)
    #
    # nerve_dir = f'nerve_dataset/nerve_images'
    # nerve_images = os.listdir(nerve_dir)

    artery_videos = [1,3,4,5,7,8,9,11,12,13,20,22,23,24,25,27,30,31,32,33,34,36,37,38]
    ureter_videos = [1,3,4,5,8,11,12,13,20,21,22,24,29,30,31,32,33,23,25]
    nerve_videos = [14,15,18,19,17]
    original_dist_artery = {}
    equal_dist_artery = {}

    original_dist_ureter = {}
    equal_dist_ureter = {}

    original_dist_nerve = {}
    equal_dist_nerve = {}
    artery_sum = 0
    ureter_sum = 0
    nerve_sum = 0

    for image in artery_images:
        if image == 'desktop.ini':
            continue
        print(image)
        title_num = image.split('.')[0].split('_')[1]
        if int(title_num) in artery_videos:
            if title_num not in original_dist_artery:
                original_dist_artery[title_num] = 24
                equal_dist_artery[title_num] = 200
            else:
                original_dist_artery[title_num] += 24
                artery_sum += 24
        if int(title_num) in ureter_videos:
            if title_num not in original_dist_ureter:
                original_dist_ureter[title_num] = 24
                equal_dist_ureter[title_num] = 240
            else:
                original_dist_ureter[title_num] += 24
                ureter_sum += 24
        if int(title_num) in nerve_videos:
            if title_num not in original_dist_nerve:
                original_dist_nerve[title_num] = 24
                equal_dist_nerve[title_num] = 1000
            else:
                original_dist_nerve[title_num] += 24
                nerve_sum += 24

    print(artery_sum, ureter_sum, nerve_sum)
    original_dist_artery = collections.OrderedDict(sorted(original_dist_artery.items(), key= lambda x: int(x[0])))
    equal_dist_artery = collections.OrderedDict(sorted(equal_dist_artery.items(), key= lambda x: int(x[0])))



    original_dist_ureter = collections.OrderedDict(sorted(original_dist_ureter.items(), key= lambda x: int(x[0])))
    equal_dist_ureter = collections.OrderedDict(sorted(equal_dist_ureter.items(), key= lambda x: int(x[0])))


    original_dist_nerve = collections.OrderedDict(sorted(original_dist_nerve.items(), key= lambda x: int(x[0])))
    equal_dist_nerve = collections.OrderedDict(sorted(equal_dist_nerve.items(), key= lambda x: int(x[0])))

    fig,ax = plt.subplots(1,3)
    ax[0].bar(original_dist_artery.keys(), original_dist_artery.values(), color = 'red')
    ax[0].axhline(y = 200, linestyle = '--')
    ax[0].set_xlabel('Video number', fontdict={'size': 14})
    ax[0].set_ylabel('Frames per video', fontdict={'size': 14})
    ax[0].set_title('Uterine artery dataset', fontdict={'size': 18})

    # ax[1,0].bar(equal_dist_artery.keys(), equal_dist_artery.values(), color = 'red')
    # ax[1,0].set_xlabel('Video number', fontdict={'size': 14})
    # ax[1,0].set_ylabel('Frames per video', fontdict={'size': 14})
    # ax[1,0].set_title('Uterine artery dataset\nequal distribution', fontdict={'size': 18})
    # ax[1,0].set_ylim(0,1200)

    ax[1].bar(original_dist_ureter.keys(), original_dist_ureter.values(), color = 'green')
    ax[1].axhline(y=240, linestyle='--')
    ax[1].set_xlabel('Video number', fontdict={'size': 14})
    ax[1].set_ylabel('Frames per video', fontdict={'size': 14})
    ax[1].set_title('Ureter dataset', fontdict={'size': 18})

    # ax[1,1].bar(equal_dist_ureter.keys(), equal_dist_ureter.values(), color = 'green')
    # ax[1,1].set_xlabel('Video number', fontdict={'size': 14})
    # ax[1,1].set_ylabel('Frames per video', fontdict={'size': 14})
    # ax[1,1].set_title('Ureter dataset\nequal distribution', fontdict={'size': 18})
    # ax[1,1].set_ylim(0,1200)

    ax[2].bar(original_dist_nerve.keys(), original_dist_nerve.values())
    ax[2].axhline(y=1000, linestyle='--')
    ax[2].set_xlabel('Video number', fontdict={'size': 14})
    ax[2].set_ylabel('Frames per video', fontdict={'size': 14})
    ax[2].set_title('Nerve dataset', fontdict={'size': 18})

    # ax[1,2].bar(equal_dist_nerve.keys(), equal_dist_nerve.values())
    # ax[1,2].set_xlabel('Video number', fontdict={'size': 14})
    # ax[1,2].set_ylabel('Frames per video', fontdict={'size': 14})
    # ax[1,2].set_title('Nerve dataset\nequal distribution', fontdict={'size': 18})
    # ax[1,2].set_ylim(0,1200)
    plt.subplots_adjust(hspace=0.5)
    plt.show()

def plot_metrics():
    import numpy as np
    # x_values = ['Pixel-wise\nensemble', 'Weighted Pixel-wise\nensemble',
    #             'Region-based\nensemble', 'Weighted Region-based\nensemble']
    # y_values = [0.3012,0.4999,0.3645,0.3681]
    #
    #
    # plt.title('Ureter Dice coefficient', fontdict={'size': 20})
    # plt.bar(x_values, y_values, color = 'blue', width=0.15)
    # plt.plot(x_values, y_values, color='green', marker = 'o')
    # #plt.axhline(y = 0.3886, linestyle = 'dashed', color = 'red')
    # plt.show()

    labels = ['Uterine artery', 'Ureter', 'Nerve', 'Image']
    multiclass = [0.0514, 0.3886, 0.0997, 0.8252]
    pixel = [0.3849, 0.5522, 0.5342, 0.9419]
    pixel_weighted = [0.3824, 0.5609, 0.5342, 0.9429]
    region = [0.6775, 0.6346, 0.5685, 0.9429]
    region_weighted = [0.6775, 0.6347, 0.5612, 0.9492]

    x = np.arange(len(labels))  # the label locations
    width = 0.15 # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 2 * width, multiclass, width, label='Multiclass', color = (0,0,0))
    rects2 = ax.bar(x - width, pixel, width, label='Pixel-wise', color = (39/255, 25/255, 191/255))
    rects3 = ax.bar(x, pixel_weighted, width, label='Weighted Pixel-wise', color = (89/255, 79/255, 194/255))
    rects4 = ax.bar(x + width, region, width, label='Region-based', color = (133/255, 128/255, 194/ 255))
    rects5 = ax.bar(x + 2 * width, region_weighted, width, label='Weighted Region-based', color = (174/255, 173/255, 186/ 255))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Dice score')
    ax.set_title('Dice scores (Equal dataset distribution)')

    ax.set_xticks(x, labels)
    ax.legend()
    plt.ylim([0,1])

    #ax.bar_label(rects1, padding=3)
    #ax.bar_label(rects2, padding=3)
    #ax.bar_label(rects3, padding=3)
    #ax.bar_label(rects4, padding=3)
    #ax.bar_label(rects5, padding=3)

    #fig.tight_layout()

    plt.show()

plot_metrics()

