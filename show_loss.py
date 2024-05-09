import matplotlib.pyplot as plt

loss = []
step = []
# file_path = './model_two/loss.txt'
file_path = './model_multi/loss.txt'
t = 0
with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        t = t + 1
        data_line = line.strip("\n").split()  # 去除首尾换行符，并按空格划分
        print("data_line", data_line[0])
        loss.append(float(data_line[0]))
        step.append(t)

print("loss", loss)
print("len loss", len(loss))

plt.figure(figsize=(10, 7))
plt.plot(step, loss)
plt.title('loss')
plt.xlabel('step')
plt.ylabel('loss')
plt.show()
