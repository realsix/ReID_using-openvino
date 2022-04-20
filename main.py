from ReID_demo import run_demo


def second_to_hour(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return [int(h), int(m), int(s)]


def hour_to_second(h, m, s):
    return (h * 60 + m) * 60 + s


if __name__ == '__main__':
    raw_message = run_demo()
    information_map = {0: [30000, 'laboratory'], 1: [20000, 'supermarket']}
    fps = 30
    message = []

    for i in raw_message:
        label = i[0]
        cam_id = i[1]
        cam_time = i[2][0]
        appear_in_second = information_map[cam_id][0] + (cam_time / fps)
        appear_in_hour = second_to_hour(appear_in_second)
        location = information_map[cam_id][1]
        h, m, s = appear_in_hour
        string = "label{} appeared at {}:{}:{} in {}".format(label, h, m, s, location)
        message.append(string + "\n")
        print(string)
    with open('tracking.txt', 'w') as txt:
        txt.writelines(message)




