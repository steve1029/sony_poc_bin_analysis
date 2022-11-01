import cv2, imageio
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.constants import c
import numpy as np
import open3d as o3d
from os import listdir
from os.path import isfile, join


"""Produce pcd data from the 8-bit png file which is extracted
 from the bin file by Sony POC.
"""

class BinToDepth:

    def __init__(self, loc, mode) -> None:
        """List depth file in the directory.

        Parameters
        ----------
        loc: str
            A path of a directory.

        Returns
        -------
        None
        """

        self.mode = mode
        onlyfiles = [f for f in listdir(loc) if isfile(join(loc, f))]
        self.onlydepth = sorted([depth for depth in onlyfiles if 'Depth_raw' in depth])
        self.imgs = []

        self.btd = c * 1e-9 * .5 # bin to distance. TDC counts signal per 1 ns.

        for im_name in self.onlydepth:

            im = cv2.imread(join(loc, im_name), cv2.IMREAD_UNCHANGED)
            self.imgs.append(im)

    def _read_area(self, srt, end):
        """Return the depth of the given area.

        Parameters
        ----------
        srt: tuple.
            the ij index of the upper left corner.

        end: tuple.
            the ij index of the lower right corner.

        Returns
        -------
        area_depth: ndarray.
            the depth value of the given area.
        """

        self.area_depth = []

        for img in self.imgs:
            self.area_depth.append(img[srt[0]:end[0], srt[1]:end[1]])

        return self.area_depth

    def _get_data(self, srt, end):

        areas = self._read_area(srt, end)

        allframe = np.array(areas)
        allframe_avg = np.mean(allframe)
        allframe_var = np.var(allframe, ddof=1)
        allframe_std = np.std(allframe, ddof=1)

        mins = []
        maxs = []
        avgs = []
        mfes = []

        for frame in areas:

            mfe = np.argmax(np.bincount(frame.flat)) # most frequent element.

            mins.append(np.min(frame))
            maxs.append(np.max(frame))
            avgs.append(np.mean(frame))
            mfes.append(mfe)

        return mins, maxs, avgs, mfes

    def _extra_axis(self, axis, x, y):

        extra_axis = matplotlib.axis.YAxis(axis)
        extra_axis.tick_right()
        #print(axis.get_yticks()*self.btd)
        yticks = axis.get_yticks()
        extra_axis.set_ticks(yticks)
        #yticklabels = [].append('%5.2f' %F for F in [yticks*self.btd])
        yticklabels = ['%5.2f' %F for F in yticks*self.btd]
        extra_axis.set_ticklabels(yticklabels)
        extra_axis.set_label_position('right')
        extra_axis.set_offset_position('right')
        extra_axis.set_label_text('Distance (m)')
        extra_axis.set_label_coords(x, y)
        #extra_axis.ylabel('Distance (m)', labelpad=775)

        return extra_axis

    def subplots(self) -> None:

        # Target at 10m with reflectivity of 3,20 %.
        tg_10m_r03_ul = (29, 126)  # upper left corner.
        tg_10m_r03_lr = (31, 145)  # lower right corner.
        tg_10m_r20_ul = (64, 126)  # upper left corner.
        tg_10m_r20_lr = (70, 147)  # lower right corner.

        # Target at 20m with reflectivity of 3,20,65,95 %.
        tg_20m_r03_ul = (33, 212)  # upper left corner.
        tg_20m_r03_lr = (45, 221)  # lower right corner.
        tg_20m_r20_ul = (53, 211)  # upper left corner.
        tg_20m_r20_lr = (59, 221)  # lower right corner.
        tg_20m_r65_ul = (38, 376)  # upper left corner.
        tg_20m_r65_lr = (46, 386)  # lower right corner.
        tg_20m_r95_ul = (38, 395)  # upper left corner.
        tg_20m_r95_lr = (45, 404)  # lower right corner.

        # Target at 35m with reflectivity of 3,20,65,95 %.
        tg_35m_r03_ul = (39, 256)  # upper left corner.
        tg_35m_r03_lr = (54, 261)  # lower right corner.
        tg_35m_r20_ul = (39, 340)  # upper left corner.
        tg_35m_r20_lr = (47, 345)  # lower right corner.
        tg_35m_r65_ul = (37, 268)  # upper left corner.
        tg_35m_r65_lr = (45, 271)  # lower right corner.
        tg_35m_r95_ul = (41, 351)  # upper left corner.
        tg_35m_r95_lr = (46, 355)  # lower right corner.

        #depth = loadder.imgs
        #tg_10m_r03 = loadder._read_area(tg_10m_r03_ul, tg_10m_r03_lr)
        #tg_10m_r20 = loadder._read_area(tg_10m_r20_ul, tg_10m_r20_lr)
        #tg_20m_r03 = loadder._read_area(tg_20m_r03_ul, tg_20m_r03_lr)
        #tg_20m_r65 = loadder._read_area(tg_20m_r65_ul, tg_20m_r65_lr)
        #tg_20m_r20 = loadder._read_area(tg_20m_r20_ul, tg_20m_r20_lr)
        #tg_20m_r95 = loadder._read_area(tg_20m_r95_ul, tg_20m_r95_lr)
        #tg_35m_r03 = loadder._read_area(tg_35m_r03_ul, tg_35m_r03_lr)
        #tg_35m_r65 = loadder._read_area(tg_35m_r65_ul, tg_35m_r65_lr)
        #tg_35m_r20 = loadder._read_area(tg_35m_r20_ul, tg_35m_r20_lr)
        #tg_35m_r95 = loadder._read_area(tg_35m_r95_ul, tg_35m_r95_lr)
        
        tg_10m_r03_mins, tg_10m_r03_maxs, tg_10m_r03_avgs, tg_10m_r03_mfes = loadder._get_data(tg_10m_r03_ul, tg_10m_r03_lr)
        tg_10m_r20_mins, tg_10m_r20_maxs, tg_10m_r20_avgs, tg_10m_r20_mfes = loadder._get_data(tg_10m_r20_ul, tg_10m_r20_lr)

        tg_20m_r03_mins, tg_20m_r03_maxs, tg_20m_r03_avgs, tg_20m_r03_mfes = loadder._get_data(tg_20m_r03_ul, tg_20m_r03_lr)
        tg_20m_r20_mins, tg_20m_r20_maxs, tg_20m_r20_avgs, tg_20m_r20_mfes = loadder._get_data(tg_20m_r20_ul, tg_20m_r20_lr)
        tg_20m_r65_mins, tg_20m_r65_maxs, tg_20m_r65_avgs, tg_20m_r65_mfes = loadder._get_data(tg_20m_r65_ul, tg_20m_r65_lr)
        tg_20m_r95_mins, tg_20m_r95_maxs, tg_20m_r95_avgs, tg_20m_r95_mfes = loadder._get_data(tg_20m_r95_ul, tg_20m_r95_lr)

        tg_35m_r03_mins, tg_35m_r03_maxs, tg_35m_r03_avgs, tg_35m_r03_mfes = loadder._get_data(tg_35m_r03_ul, tg_35m_r03_lr)
        tg_35m_r20_mins, tg_35m_r20_maxs, tg_35m_r20_avgs, tg_35m_r20_mfes = loadder._get_data(tg_35m_r20_ul, tg_35m_r20_lr)
        tg_35m_r65_mins, tg_35m_r65_maxs, tg_35m_r65_avgs, tg_35m_r65_mfes = loadder._get_data(tg_35m_r65_ul, tg_35m_r65_lr)
        tg_35m_r95_mins, tg_35m_r95_maxs, tg_35m_r95_avgs, tg_35m_r95_mfes = loadder._get_data(tg_35m_r95_ul, tg_35m_r95_lr)

        nframes = np.arange(len(tg_10m_r03_maxs))
        fig10, axes10 = plt.subplots(nrows=1, ncols=2, figsize=(12,4), frameon=True)
        fig20, axes20 = plt.subplots(nrows=1, ncols=4, figsize=(24,4), frameon=True)
        fig35, axes35 = plt.subplots(nrows=1, ncols=4, figsize=(24,4), frameon=True)

        #fig.supxlabel('Frames')
        #fig.supylabel('Distance (m)')
        axes10[0].plot(nframes, tg_10m_r03_mins, label='min')
        axes10[0].plot(nframes, tg_10m_r03_maxs, label='max')
        axes10[0].plot(nframes, tg_10m_r03_avgs, label='avgs')
        axes10[0].plot(nframes, tg_10m_r03_mfes, label='mfe')
        axes10[1].plot(nframes, tg_10m_r20_mins, label='min')
        axes10[1].plot(nframes, tg_10m_r20_maxs, label='max')
        axes10[1].plot(nframes, tg_10m_r20_avgs, label='avgs')
        axes10[1].plot(nframes, tg_10m_r20_mfes, label='mfe')

        axes20[0].plot(nframes, tg_20m_r03_mins, label='min')
        axes20[0].plot(nframes, tg_20m_r03_maxs, label='max')
        axes20[0].plot(nframes, tg_20m_r03_avgs, label='avgs')
        axes20[0].plot(nframes, tg_20m_r03_mfes, label='mfe')
        axes20[1].plot(nframes, tg_20m_r20_mins, label='min')
        axes20[1].plot(nframes, tg_20m_r20_maxs, label='max')
        axes20[1].plot(nframes, tg_20m_r20_avgs, label='avgs')
        axes20[1].plot(nframes, tg_20m_r20_mfes, label='mfe')
        axes20[2].plot(nframes, tg_20m_r65_mins, label='min')
        axes20[2].plot(nframes, tg_20m_r65_maxs, label='max')
        axes20[2].plot(nframes, tg_20m_r65_avgs, label='avgs')
        axes20[2].plot(nframes, tg_20m_r65_mfes, label='mfe')
        axes20[3].plot(nframes, tg_20m_r95_mins, label='min')
        axes20[3].plot(nframes, tg_20m_r95_maxs, label='max')
        axes20[3].plot(nframes, tg_20m_r95_avgs, label='avgs')
        axes20[3].plot(nframes, tg_20m_r95_mfes, label='mfe')

        axes35[0].plot(nframes, tg_35m_r03_mins, label='min')
        axes35[0].plot(nframes, tg_35m_r03_maxs, label='max')
        axes35[0].plot(nframes, tg_35m_r03_avgs, label='avgs')
        axes35[0].plot(nframes, tg_35m_r03_mfes, label='mfe')
        axes35[1].plot(nframes, tg_35m_r20_mins, label='min')
        axes35[1].plot(nframes, tg_35m_r20_maxs, label='max')
        axes35[1].plot(nframes, tg_35m_r20_avgs, label='avgs')
        axes35[1].plot(nframes, tg_35m_r20_mfes, label='mfe')
        axes35[2].plot(nframes, tg_35m_r65_mins, label='min')
        axes35[2].plot(nframes, tg_35m_r65_maxs, label='max')
        axes35[2].plot(nframes, tg_35m_r65_avgs, label='avgs')
        axes35[2].plot(nframes, tg_35m_r65_mfes, label='mfe')
        axes35[3].plot(nframes, tg_35m_r95_mins, label='min')
        axes35[3].plot(nframes, tg_35m_r95_maxs, label='max')
        axes35[3].plot(nframes, tg_35m_r95_avgs, label='avgs')
        axes35[3].plot(nframes, tg_35m_r95_mfes, label='mfe')

        axes10[0].legend(loc='best', prop={'size': 10})
        axes20[0].legend(loc='best', prop={'size': 10})
        axes35[0].legend(loc='best', prop={'size': 10})
        """
        axes[0,1].legend(loc='best', prop={'size': 10})
        axes[0,2].legend(loc='best', prop={'size': 10})
        axes[1,0].legend(loc='best', prop={'size': 10})
        axes[1,1].legend(loc='best', prop={'size': 10})
        axes[1,2].legend(loc='best', prop={'size': 10})
        """

        axes10[0].set_title('10m, 3% Ref')
        axes10[1].set_title('10m, 20% Ref')
        axes20[0].set_title('20m, 3% Ref')
        axes20[1].set_title('20m, 20% Ref')
        axes20[2].set_title('20m, 65% Ref')
        axes20[3].set_title('20m, 95% Ref')
        axes35[0].set_title('35m, 3% Ref')
        axes35[1].set_title('35m, 20% Ref')
        axes35[2].set_title('35m, 65% Ref')
        axes35[3].set_title('35m, 95% Ref')

        axes10[0].set_xlabel('Frames')
        axes10[1].set_xlabel('Frames')
        axes20[0].set_xlabel('Frames')
        axes20[1].set_xlabel('Frames')
        axes20[2].set_xlabel('Frames')
        axes20[3].set_xlabel('Frames')
        axes35[0].set_xlabel('Frames')
        axes35[1].set_xlabel('Frames')
        axes35[2].set_xlabel('Frames')
        axes35[3].set_xlabel('Frames')

        """
        handles10, labels10 = axes10[0].get_legend_handles_labels()
        handles35, labels35 = axes35[0].get_legend_handles_labels()
        fig10.legend(handles10, labels10, loc='center left', prop={'size':12}, framealpha=0)
        fig35.legend(handles35, labels35, loc='center left', prop={'size':12}, framealpha=0)
        """

        #print(np.min(tg_10m_r03_mins+tg_10m_r20_mins))
        ylim_10m = (np.min(tg_10m_r03_mins+tg_10m_r20_mins)*0.99, np.max(tg_10m_r03_maxs+tg_10m_r20_maxs)*1.01)

        ylim_20m = (np.min(tg_20m_r03_mins+tg_20m_r20_mins+tg_20m_r65_mins+tg_20m_r95_mins)*0.995,\
            np.max(tg_20m_r03_maxs+tg_20m_r20_maxs+tg_20m_r65_maxs+tg_20m_r95_maxs)*1.005)

        ylim_35m = (np.min(tg_35m_r03_mins+tg_35m_r20_mins+tg_35m_r65_mins+tg_35m_r95_mins)*0.998,\
            np.max(tg_35m_r03_maxs+tg_35m_r20_maxs+tg_35m_r65_maxs+tg_35m_r95_maxs)*1.002)

        axes10[0].set_ylim(ylim_10m)
        axes10[1].set_ylim(ylim_10m)
        axes20[0].set_ylim(ylim_20m)
        axes20[1].set_ylim(ylim_20m)
        axes20[2].set_ylim(ylim_20m)
        axes20[3].set_ylim(ylim_20m)
        axes35[0].set_ylim(ylim_35m)
        axes35[1].set_ylim(ylim_35m)
        axes35[2].set_ylim(ylim_35m)
        axes35[3].set_ylim(ylim_35m)

        axes10[0].set_ylabel('bin')
        axes10[1].set_ylabel('bin')
        axes20[0].set_ylabel('bin')
        axes20[1].set_ylabel('bin')
        axes20[2].set_ylabel('bin')
        axes20[3].set_ylabel('bin')
        axes35[0].set_ylabel('bin')
        axes35[1].set_ylabel('bin')
        axes35[2].set_ylabel('bin')
        axes35[3].set_ylabel('bin')

        axes10[0].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes10[1].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes20[0].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes20[1].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes20[2].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes20[3].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes35[0].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes35[1].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes35[2].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes35[3].yaxis.set_major_locator(MaxNLocator(integer=True))

        #fig10.savefig('./bin_min_max_10m.png', dpi=300, bbox_inches='tight', transparent=True)
        #fig35.savefig('./bin_min_max_35m.png', dpi=300, bbox_inches='tight', transparent=True)

        #fig10.savefig('./depth_min_max_10m.png', dpi=300, bbox_inches='tight', transparent=True)
        #fig35.savefig('./depth_min_max_35m.png', dpi=300, bbox_inches='tight', transparent=True)

        """
        eaxes100 = axes10[0].twinx()
        eaxes100.yaxis.set_label_position('right')
        eaxes100.set_yticks(axes10[0].get_yticks())
        eaxes100.set_yticklabels(axes10[0].get_yticks*self.btd)

        """

        x = 1.15
        y = 0.5
        eaxis100 = self._extra_axis(axes10[0], x, y)
        eaxis101 = self._extra_axis(axes10[1], x, y)

        eaxis200 = self._extra_axis(axes20[0], x, y)
        eaxis201 = self._extra_axis(axes20[1], x, y)
        eaxis202 = self._extra_axis(axes20[2], x, y)
        eaxis203 = self._extra_axis(axes20[3], x, y)

        eaxis350 = self._extra_axis(axes35[0], x, y)
        eaxis351 = self._extra_axis(axes35[1], x, y)
        eaxis352 = self._extra_axis(axes35[2], x, y)
        eaxis353 = self._extra_axis(axes35[3], x, y)

        axes10[0].add_artist(eaxis100)
        axes10[1].add_artist(eaxis101)
        axes20[0].add_artist(eaxis200)
        axes20[1].add_artist(eaxis201)
        axes20[2].add_artist(eaxis202)
        axes20[3].add_artist(eaxis203)
        axes35[0].add_artist(eaxis350)
        axes35[1].add_artist(eaxis351)
        axes35[2].add_artist(eaxis352)
        axes35[3].add_artist(eaxis353)

        #axes10[1].set_ylabel('Distance (m)')
        #axes35[0].set_ylabel('Distance (m)')
        #axes35[1].set_ylabel('Distance (m)')
        #axes35[2].set_ylabel('Distance (m)')
        #axes35[3].set_ylabel('Distance (m)')
        fig10.subplots_adjust(wspace=0.4)
        fig20.subplots_adjust(wspace=0.4)
        fig35.subplots_adjust(wspace=0.4)
        fig10.savefig(f'./{self.mode}_bin_depth_10m.png', dpi=300, bbox_inches='tight', transparent=True)
        fig20.savefig(f'./{self.mode}_bin_depth_20m.png', dpi=300, bbox_inches='tight', transparent=True)
        fig35.savefig(f'./{self.mode}_bin_depth_35m.png', dpi=300, bbox_inches='tight', transparent=True)

        #ax00 = axes[0,0].twinx()
        #ax00.set_ylabel('bin')
        #ax00.plot(nframes, tg_10m_r03_mins_dist, label='min')
        #ax00.plot(nframes, tg_10m_r03_maxs_dist, label='max')
        #ax00.tick_params(axis='y')

        return


if __name__ == '__main__':

    loc = 'D:/sony_poc_range96_paju_image_10s/'
    mode = 'range96'
    loc = 'D:/sony_poc_range192_paju_image_10s/'
    mode = 'range192'
    loc = 'D:/sony_poc_range192_paju_image_10s/'
    mode = 'range192'

    loadder = BinToDepth(loc, mode)
    loadder.subplots()