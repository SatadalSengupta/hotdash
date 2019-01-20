import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import itertools
from math import log10

RESULTS_FOLDER = './results_adjusted/'
NUM_BINS = 100
BITS_IN_BYTE = 8.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
# VIDEO_LEN = 64
## Custom start
# VIDEO_LEN = 49
VIDEO_LEN = 48
## Custom end
VIDEO_BIT_RATE = [350, 600, 1000, 2000, 3000]
COLOR_MAP = plt.cm.jet #nipy_spectral, Set1,Paired 
SIM_DP = 'sim_dp'
# SCHEMES = ['BB', 'RB', 'FIXED', 'FESTIVE', 'BOLA', 'RL', 'robustMPC', 'fastMPC', 'sim_rl', SIM_DP]
# SCHEMES = ['BB', 'FESTIVE', 'BOLA', 'RL', 'robustMPC', 'fastMPC']
SCHEMES = ['BB', 'RB', 'FESTIVE', 'RL', 'robustMPC', 'fastMPC', 'HotDASH']
# SCHEMES = ['BB', 'RB', 'FESTIVE', 'RL', 'robustMPC', 'fastMPC']

##################################################

def generate_cdf(data_points):
    """ Returns empirical CDF for provided set of data points """

    freq_dist = Counter(data_points)
    xvals = sorted(freq_dist.keys())

    #pos_nz = 0
    
    #for i, xval in enumerate(xvals):
    #    if xval > 0:
    #        pos_nz = i
    #        break
    #xvals = xvals[pos_nz:]

    ph_xvals = [xval+(1-xvals[0]) for xval in xvals]

    plot_ph_xvals = np.logspace(start=log10(ph_xvals[0]), stop=log10(ph_xvals[-1]), num=100, base=10)
    plot_xvals = [xval+xvals[0]-1 for xval in plot_ph_xvals]
    #print plot_xvals
    plot_yvals = []

    cum_freq = 0
    last_pos = 0

    for plot_xval in plot_xvals:
        for xval in xvals[last_pos:]:
            if xval > plot_xval:
                break
            cum_freq += freq_dist[xval]
            last_pos += 1
        plot_yvals.append(cum_freq/float(len(data_points)))

    return plot_xvals, plot_yvals

##################################################

def scheme_to_label(scheme):
    	label = scheme
	if scheme == "BB":
    		label = "BB"
	elif scheme == "RB":
    		label = "RB"
	elif scheme == "FIXED":
    		label = "Fixed"
	elif scheme == "RL":
    		label = "Pensieve"
	elif scheme == "FESTIVE":
    		label = "Festive"
	elif scheme == "BOLA":
    			label = "BOLA"
	elif scheme == "robustMPC":
    			label = "robustMPC"
	elif scheme == "fastMPC":
    			label = "fastMPC"
	elif scheme == "HotDASH":
    		label = "HotDASH"
	return label + ": {:.2f}"

def main():
	time_all = {}
	bit_rate_all = {}
	buff_all = {}
	bw_all = {}
	raw_reward_all = {}

	for scheme in SCHEMES:
		time_all[scheme] = {}
		raw_reward_all[scheme] = {}
		bit_rate_all[scheme] = {}
		buff_all[scheme] = {}
		bw_all[scheme] = {}

	log_files = os.listdir(RESULTS_FOLDER)
	# print "count: {}".format(len(log_files))
	for log_file in log_files:

		time_ms = []
		bit_rate = []
		buff = []
		bw = []
		reward = []

		# print log_file
		with open(RESULTS_FOLDER + log_file, 'rb') as f:
			# print "file {}".format(RESULTS_FOLDER + log_file)
			lines = f.readlines()
			len_file = len(lines)
			# print len_file
			if len_file < VIDEO_LEN:
				continue

			if SIM_DP in log_file:
				# print "here"
				for line in f:
					parse = line.split()
					if len(parse) == 1:
						reward = float(parse[0])
					elif len(parse) >= 6:
						time_ms.append(float(parse[3]))
						bit_rate.append(VIDEO_BIT_RATE[int(parse[6])])
						buff.append(float(parse[4]))
						bw.append(float(parse[5]))

			else:
				for line in lines:
					parse = line.split()
					# print len(parse)
					if len(parse) <= 1:
    						# print "break: {}".format(line)
						break
					try:
						time_ms.append(float(parse[0]))
						bit_rate.append(int(parse[1]))
						buff.append(float(parse[2]))
						bw.append(float(parse[4]) / float(parse[5]) * BITS_IN_BYTE * MILLISEC_IN_SEC / M_IN_B)
						reward.append(float(parse[6]))
					except Exception as e:
						print "Parse error: {}".format(log_file)
						continue
				## Custom addition
				# reward.append(78.5)

		if SIM_DP in log_file:
			time_ms = time_ms[::-1]
			bit_rate = bit_rate[::-1]
			buff = buff[::-1]
			bw = bw[::-1]
		
		time_ms = np.array(time_ms)
		time_ms -= time_ms[0]
		
		# print log_file

		for scheme in SCHEMES:
			if scheme in log_file:
				time_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = time_ms
				bit_rate_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bit_rate
				buff_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = buff
				bw_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bw
				raw_reward_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = reward
				break

	## Custom start
	# for scheme in SCHEMES:
	# 	print scheme + " " + str(len(time_all[scheme].keys()))
	## Custom end

	# ---- ---- ---- ----
	# Reward records
	# ---- ---- ---- ----
		
	log_file_all = []
	reward_all = {}
	for scheme in SCHEMES:
		reward_all[scheme] = []

	for l in time_all[SCHEMES[0]]:
		schemes_check = True
		for scheme in SCHEMES:
			if l not in time_all[scheme] or len(time_all[scheme][l]) < VIDEO_LEN:
				schemes_check = False
				## The problem is here!!!
				## Custom start
				# if l not in time_all[scheme]:
    			# 		print "l: {}".format(l)
				# 	print "time_all[scheme]: {}".format(time_all[scheme])
				# elif len(time_all[scheme][l]) < VIDEO_LEN:
    			# 		print "len(time_all[scheme][l]): {}".format(len(time_all[scheme][l]))
				# 	print "VIDEO_LEN: {}".format(VIDEO_LEN)
				# 	print "scheme: {}; l: {}\n".format(scheme, l)
				## Custom end
				break
		if schemes_check:
			log_file_all.append(l)
			for scheme in SCHEMES:
				if scheme == SIM_DP:
					reward_all[scheme].append(raw_reward_all[scheme][l])
				else:
					reward_all[scheme].append(np.sum(raw_reward_all[scheme][l][1:VIDEO_LEN]))
					if scheme == 'HotDASH' and np.sum(raw_reward_all[scheme][l][1:VIDEO_LEN]) < 0:
    						print l

	## Custom start
	# for scheme in SCHEMES:
    # 		print "Reward " + scheme + ": " + str(len(reward_all[scheme]))
	## Custom end

	mean_rewards = {}
	for scheme in SCHEMES:
		mean_rewards[scheme] = np.mean(reward_all[scheme])

	sns.set(style="ticks", font_scale=1.4)
	marker = itertools.cycle(("o", "v", "p", "d", "h", "s", "^"))
	linestyle = itertools.cycle(("-", "-.", "--", ":"))

	fig = plt.figure()
	ax = fig.add_subplot(111)

	## Custom comment start
	# for scheme in SCHEMES:
	# 	ax.plot(reward_all[scheme])
	## Custom comment end

	## Custom start
	for scheme in SCHEMES:
    		ax.plot(reward_all[scheme], label=scheme_to_label(scheme).format(mean_rewards[scheme]),
			 linestyle=linestyle.next(), linewidth=2, marker=marker.next(), markersize=5)
	## Custom end
	
	## Custom comment start
	SCHEMES_REW = []
	for scheme in SCHEMES:
		SCHEMES_REW.append(scheme + ': ' + str(mean_rewards[scheme]))

	# colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
	# for i,j in enumerate(ax.lines):
	# 	j.set_color(colors[i])

	# ax.legend(SCHEMES_REW, loc=4)
	## Custom comment end

	## Custom start
	ax.set_xlim(0, len(reward_all['RL'])+5)
	ax.legend(loc=4)
	## Custom end
	
	plt.ylabel('total reward')
	plt.xlabel('trace index')
	# plt.show()
	plt.tight_layout()
	plt.savefig("plots/all_rewards.pdf", format="pdf")
	plt.savefig("plots/all_rewards.png", format="png")

	# ---- ---- ---- ----
	# CDF 
	# ---- ---- ---- ----

	## Custom start
	marker = itertools.cycle(("o", "v", "p", "d", "h", "s", "^"))
	linestyle = itertools.cycle(("-", "-.", "--", ":"))
	xlim = 0
	## Custom end

	fig = plt.figure()
	ax = fig.add_subplot(111)

	for scheme in SCHEMES:
		# values, base = np.histogram(reward_all[scheme], bins=80, density=1)
		# cumulative = np.cumsum(values)
		# ax.plot(base[:-1], cumulative)
		xvals, yvals = generate_cdf(reward_all[scheme])
		ax.plot(xvals, yvals, label=scheme_to_label(scheme).format(mean_rewards[scheme]),
			 linestyle=linestyle.next(), linewidth=2, marker=marker.next(), markersize=5)
		#xlim = len(xvals)

	## Custom comment start
	#colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
	#for i,j in enumerate(ax.lines):
	#	j.set_color(colors[i])	

	#ax.legend(SCHEMES_REW, loc=4)
	## Custom comment end

	## Custom start
	ax.legend(loc=2)
	ax.set_xlim(-500, 100)
	## Custom ends
	
	plt.ylabel('CDF')
	plt.xlabel('total reward')
	# plt.show()
	plt.tight_layout()
	plt.savefig("plots/cdf_rewards.pdf", format="pdf")
	plt.savefig("plots/cdf_rewards.png", format="png")


	# ---- ---- ---- ----
	# check each trace
	# ---- ---- ---- ----
	# sns.set(style="ticks", font_scale=0.8)

	# for l in time_all[SCHEMES[0]]:
	# 	schemes_check = True
	# 	for scheme in SCHEMES:
	# 		if l not in time_all[scheme] or len(time_all[scheme][l]) < VIDEO_LEN:
	# 			schemes_check = False
	# 			break
	# 	if schemes_check:
	# 		fig = plt.figure()

	# 		ax = fig.add_subplot(311)
	# 		for scheme in SCHEMES:
	# 			ax.plot(time_all[scheme][l][:VIDEO_LEN], bit_rate_all[scheme][l][:VIDEO_LEN])
	# 		colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
	# 		for i,j in enumerate(ax.lines):
	# 			j.set_color(colors[i])	
	# 		plt.title(l)
	# 		plt.ylabel('bit rate selection (kbps)')

	# 		ax = fig.add_subplot(312)
	# 		for scheme in SCHEMES:
	# 			ax.plot(time_all[scheme][l][:VIDEO_LEN], buff_all[scheme][l][:VIDEO_LEN])
	# 		colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
	# 		for i,j in enumerate(ax.lines):
	# 			j.set_color(colors[i])	
	# 		plt.ylabel('buffer size (sec)')

	# 		ax = fig.add_subplot(313)
	# 		for scheme in SCHEMES:
	# 			ax.plot(time_all[scheme][l][:VIDEO_LEN], bw_all[scheme][l][:VIDEO_LEN])
	# 		colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
	# 		for i,j in enumerate(ax.lines):
	# 			j.set_color(colors[i])	
	# 		plt.ylabel('bandwidth (mbps)')
	# 		plt.xlabel('time (sec)')

	# 		SCHEMES_REW = []
	# 		for scheme in SCHEMES:
	# 			if scheme == SIM_DP:
	# 				SCHEMES_REW.append(scheme + ': ' + str(raw_reward_all[scheme][l]))
	# 			else:
	# 				SCHEMES_REW.append(scheme + ': ' + str(np.sum(raw_reward_all[scheme][l][1:VIDEO_LEN])))

	# 		ax.legend(SCHEMES_REW, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=int(np.ceil(len(SCHEMES) / 2.0)))
	# 		# plt.show()
	# 		#plt.tight_layout()
	# 		plt.savefig("plots/other_params.pdf", format="pdf")
	# 		plt.savefig("plots/other_params.png", format="png")


if __name__ == '__main__':
	main()