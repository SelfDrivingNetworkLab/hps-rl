import matplotlib.pyplot as plt
x = [80,8000,16000,24000]

y1 = [93.26666667,109.3,113,93.2]
y1_max = [110,125,220,183]
y1_min = [63.8,84.9,53.9,47.1]




y2 = [38.66162733,71.89652867,77.09107,70.97540867]
y2_max = [42.734272,74.40246,87.11813,75.825966]
y2_min = [34.82061,68.56389,68.50832,62.11635]



y3 = [10.39,9.35,10.15666667,27.28]
y3_max = [11.1,10,11.1,50.3]
y3_min = [9.37,8.72,8.87,5.64]

plt.plot(x, y1, label='ACKTR')
plt.plot(x, y2, linestyle ='dashed',label='PPO')
plt.plot(x, y3, linestyle ='dotted',label='A2C')
plt.xlabel('Training Episodes')
plt.ylabel('Value Loss')
 

# alpha adjusts transparency, higher alpha --> darker grey
# Or color could be set to, for example '0.2', but using transparency allows
# overlapping shaded areas
plt.fill_between(x, y1_min, y1_max, color = 'k', alpha = 0.1)
plt.fill_between(x, y2_min, y2_max, color = 'k', alpha = 0.1)
plt.fill_between(x, y3_min, y3_max, color = 'k', alpha = 0.1)
plt.legend(loc='upper right', frameon=False)

plt.show()