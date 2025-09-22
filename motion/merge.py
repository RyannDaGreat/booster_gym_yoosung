import pickle

# Hardcoded masks
UPPER_MASK = [True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False]
LOWER_MASK = [False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True]

# Load files
with open('ys.pkl', 'rb') as f:
    ys = pickle.load(f)

with open('ys_leg.pkl', 'rb') as f:
    ys_leg = pickle.load(f)

# Sort keyframes by time
ys_keyframes = sorted(ys['puppeteer']['keyframes'], key=lambda x: x[0])
ys_leg_keyframes = sorted(ys_leg['puppeteer']['keyframes'], key=lambda x: x[0])

# Sanity check: timestamps should be equal
ys_times = [kf[0] for kf in ys_keyframes]
ys_leg_times = [kf[0] for kf in ys_leg_keyframes]
assert ys_times == ys_leg_times, f"Timestamps don't match after sorting!"

# Merge keyframes
merged_keyframes = []
for (time, ys_joints), (_, ys_leg_joints) in zip(ys_keyframes, ys_leg_keyframes):
    merged_joints = []
    for i in range(len(ys_joints)):
        if UPPER_MASK[i]:
            merged_joints.append(ys_joints[i])
        else:  # LOWER_MASK[i]
            merged_joints.append(ys_leg_joints[i])
    merged_keyframes.append((time, merged_joints))

# Save result
result = {'puppeteer': {'keyframes': merged_keyframes}}
with open('merged.pkl', 'wb') as f:
    pickle.dump(result, f)

print(f"Merged {len(merged_keyframes)} keyframes and saved to merged.pkl")