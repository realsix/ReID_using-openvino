from ReID_demo import run_demo


def second_to_hour(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return [h, m, s]


def hour_to_second(h, m, s):
    return (h * 60 + m) * 60 + s


raw_message = run_demo()

information_map = {0: [30000, 'laboratory'], 1: [20000, 'supermarket']}


fps = 60

message = []

label = 0

for i in raw_message:
    cam_id = i[0]
    cam_time = i[1]
    appear_in_second = information_map[cam_id][0] + cam_time
    appear_in_hour = second_to_hour(appear_in_second)
    location = information_map[cam_id][1]
    h, m, s = appear_in_hour
    string = "label{} appeared at {}:{}:{} in {}".format(label, h, m, s, location)
    print(string)



