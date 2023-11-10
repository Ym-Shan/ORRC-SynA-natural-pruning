firing_rate = [0.0464, 0.0448, 0.0109, 0.0250, 0.0144,
               0.0222, 0.0176, 0, 0.0244, 0.0192,
               0.0232, 0.0159, 0, 0.0211, 0.0223,
               0.0227, 0.0044, 0, 0.0123, 0.0235]

firing_rate_after_shortcut = [0.0176, 0.0159, 0.0044]

# [C, H, W]
neuron_input = [[64, 28, 28], [64, 28, 28], [64, 28, 28], [64, 28, 28], [64, 28, 28],
                [128, 14, 14], [128, 14, 14], [128, 14, 14], [128, 14, 14], [128, 14, 14],
                [256, 7, 7], [256, 7, 7], [256, 7, 7], [256, 7, 7], [256, 7, 7],
                [512, 4, 4], [512, 4, 4], [512, 4, 4], [512, 4, 4], [512, 4, 4]]

# [Cin, Cout, Kernel_size, out_size]
Conv = [[1, 64, 3, 28], [64, 64, 3, 28], [64, 64, 3, 28], [64, 64, 3, 28], [64, 64, 3, 28],
        [64, 128, 3, 14], [128, 128, 3, 14], [64, 128, 1, 14], [128, 128, 3, 14], [128, 128, 3, 14],
        [128, 256, 3, 7], [256, 256, 3, 7], [128, 256, 1, 7], [256, 256, 3, 7], [256, 256, 3, 7],
        [256, 512, 3, 4], [512, 512, 3, 4], [256, 512, 1, 4], [512, 512, 3, 4], [512, 512, 3, 4]]

linear = [512, 10]



pruning = [1,2,3]

MA = ['TA']
IA = ['TA']

# Number of CA channels in each MA
C_list_in_CA = [64, 64, 128, 128, 256, 256, 512, 512]
# Number of CA channels in each IA
C_list_in_IA_C = [128, 256, 512]
# Firing rate of LIF for TA, CA, SA in each IA
IA_T_firing_rate = [0, 0, 0]
IA_C_firing_rate = [0, 0.4961, 0.5039]
IA_S_firing_rate = [0, 0, 0]

T = 16

spike = 0
AC = 0
MAC = 0

count_multiplication = 0
count_add = 0

energy = 0

# Calculating spike
for i in range(20):
    spike += firing_rate[i] * T * neuron_input[i][0] * neuron_input[i][1] * neuron_input[i][2]

if 1 in pruning:
    spike = spike - firing_rate[7] * T * neuron_input[7][0] * neuron_input[7][1] * neuron_input[7][2]
if 2 in pruning:
    spike = spike - firing_rate[12] * T * neuron_input[12][0] * neuron_input[12][1] * neuron_input[12][2]
if 3 in pruning:
    spike = spike - firing_rate[17] * T * neuron_input[17][0] * neuron_input[17][1] * neuron_input[17][2]
# ----------------------------------------------------------------------------------------------------------------------
# Calculate AC and MAC operations
# Convolutional layer computation：(Cin * Hk * Wk - 1) * Hout * Wout * Cout
# first_block
MAC += (Conv[0][0] * Conv[0][2] * Conv[0][2] - 1) * Conv[0][3] * Conv[0][3] * Conv[0][1] * T

for i in range(1, 8):
    AC += (Conv[i][0] * Conv[i][2] * Conv[i][2] - 1) * Conv[i][3] * Conv[i][3] * Conv[i][1] * firing_rate[i-1] * T

AC += (Conv[8][0] * Conv[8][2] * Conv[8][2] - 1) * Conv[8][3] * Conv[8][3] * Conv[8][1] * firing_rate_after_shortcut[0] * T


for i in range(9, 13):
    AC += (Conv[i][0] * Conv[i][2] * Conv[i][2] - 1) * Conv[i][3] * Conv[i][3] * Conv[i][1] * firing_rate[i-1] * T

AC += (Conv[13][0] * Conv[13][2] * Conv[13][2] - 1) * Conv[13][3] * Conv[13][3] * Conv[13][1] * firing_rate_after_shortcut[1] * T


for i in range(14, 18):
    AC += (Conv[i][0] * Conv[i][2] * Conv[i][2] - 1) * Conv[i][3] * Conv[i][3] * Conv[i][1] * firing_rate[i-1] * T

AC += (Conv[18][0] * Conv[18][2] * Conv[18][2] - 1) * Conv[18][3] * Conv[18][3] * Conv[18][1] * firing_rate_after_shortcut[2] * T

AC += (Conv[19][0] * Conv[19][2] * Conv[19][2] - 1) * Conv[19][3] * Conv[19][3] * Conv[19][1] * firing_rate[18] * T

if 1 in pruning:
    AC = AC - (Conv[7][0] * Conv[7][2] * Conv[7][2] - 1) * Conv[7][3] * Conv[7][3] * Conv[7][1] * firing_rate[6] * T
if 2 in pruning:
    AC = AC - (Conv[12][0] * Conv[12][2] * Conv[12][2] - 1) * Conv[12][3] * Conv[12][3] * Conv[12][1] * firing_rate[11] * T
if 3 in pruning:
    AC = AC - (Conv[17][0] * Conv[17][2] * Conv[17][2] - 1) * Conv[17][3] * Conv[17][3] * Conv[17][1] * firing_rate[16] * T
# ----------------------------------------------------------
# BNlayer
# C * H * W
for i in range(20):
    MAC += Conv[i][1] * Conv[i][3] * Conv[i][3] * T

if 1 in pruning:
    MAC = MAC - Conv[7][1] * Conv[7][3] * Conv[7][3] * T
if 2 in pruning:
    MAC = MAC - Conv[12][1] * Conv[12][3] * Conv[12][3] * T
if 3 in pruning:
    MAC = MAC - Conv[17][1] * Conv[17][3] * Conv[17][3] * T
# ----------------------------------------------------------------------------------------------------------------------
# energy consumption
# Fully connected layer
# mul = Cin * Cout
# add = Cin * Cin
count_multiplication += linear[0] * linear[1] * T
count_add += linear[0] * linear[0] * T
# ----------------------------------------------------------
# AC and MAC
count_multiplication += MAC
count_add += MAC
count_add += AC

energy += count_multiplication * 3.7 + count_add * 0.9
#----------------------------------------------------------------------------------------------------------------------
# Additional energy consumption of attention module
# Convolutional layer computation：(Cin * Hk * Wk - 1) * Hout * Wout * Cout
if 'TA' in MA:
    MAC += (T * 1 * 1 - 1) * 1 * 1 * T / 4 * 8          # There are a total of 8 MA modules in the entire network
    MAC += (T / 4 * 1 * 1 - 1) * 1 * 1 * T * 8
if 'CA' in MA:
    for i in range(8):
        # if i != 3 and i != 5 and i != 7:
        MAC += (C_list_in_CA[i] * 1 * 1 - 1) * 1 * 1 * C_list_in_CA[i] / 8
        MAC += (C_list_in_CA[i] / 8 * 1 * 1 - 1) * 1 * 1 * C_list_in_CA[i]
if 'SA' in MA:
    MAC += (2 * 7 * 7 - 1) * 28 * 28 * 1        # The first two convolutional kernels have a size of 7
    MAC += (2 * 7 * 7 - 1) * 28 * 28 * 1
    MAC += (2 * 3 * 3 - 1) * 14 * 14 * 1
    MAC += (2 * 3 * 3 - 1) * 14 * 14 * 1
    MAC += (2 * 3 * 3 - 1) * 7 * 7 * 1
    MAC += (2 * 3 * 3 - 1) * 7 * 7 * 1
    MAC += (2 * 3 * 3 - 1) * 4 * 4 * 1
    MAC += (2 * 3 * 3 - 1) * 4 * 4 * 1
# shortcut--------------------------------------------------------
if 'TA' in IA:
    if 1 not in pruning:
        MAC += (T * 1 * 1 - 1) * 1 * 1 * T / 4
        MAC += (T / 4 * 1 * 1 - 1) * 1 * 1 * T
        spike += T * IA_T_firing_rate[0]
    if 2 not in pruning:
        MAC += (T * 1 * 1 - 1) * 1 * 1 * T / 4
        MAC += (T / 4 * 1 * 1 - 1) * 1 * 1 * T
        spike += T * IA_T_firing_rate[1]
    if 3 not in pruning:
        MAC += (T * 1 * 1 - 1) * 1 * 1 * T / 4
        MAC += (T / 4 * 1 * 1 - 1) * 1 * 1 * T
        spike += T * IA_T_firing_rate[2]

if 'CA' in IA:
    if 1 not in pruning:
        MAC += (C_list_in_IA_C[0] * 1 * 1 - 1) * 1 * 1 * C_list_in_IA_C[0] / 8
        MAC += (C_list_in_IA_C[0] / 8 * 1 * 1 - 1) * 1 * 1 * C_list_in_IA_C[0]
        spike += 128 * IA_C_firing_rate[0]
    if 2 not in pruning:
        MAC += (C_list_in_IA_C[1] * 1 * 1 - 1) * 1 * 1 * C_list_in_IA_C[1] / 8
        MAC += (C_list_in_IA_C[1] / 8 * 1 * 1 - 1) * 1 * 1 * C_list_in_IA_C[1]
        spike += 256 * IA_C_firing_rate[1]
    if 3 not in pruning:
        MAC += (C_list_in_IA_C[2] * 1 * 1 - 1) * 1 * 1 * C_list_in_IA_C[2] / 8
        MAC += (C_list_in_IA_C[2] / 8 * 1 * 1 - 1) * 1 * 1 * C_list_in_IA_C[2]
        spike += 512 * IA_C_firing_rate[2]

if 'SA' in IA:
    if 1 not in pruning:
        MAC += (2 * 3 * 3 - 1) * 14 * 14 * 1
        spike += 4 * 4 * IA_S_firing_rate[0]
    if 2 not in pruning:
        MAC += (2 * 3 * 3 - 1) * 7 * 7 * 1
        spike += 2 * 2 * IA_S_firing_rate[1]
    if 3 not in pruning:
        MAC += (2 * 3 * 3 - 1) * 4 * 4 * 1
        spike += 1 * 1 * IA_S_firing_rate[2]

print("spike数量：", end=' ')
print(spike / 1000, end=' ')
print('K')
print("AC运算量：", end=' ')
print(AC / 1000000, end=' ')
print("Millon")
print("MAC运算量：", end=' ')
print(MAC / 1000000, end=' ')
print("Millon")
print("能耗为：", end=' ')
print(energy * 0.000000001, end=' ')
print('mJ')

