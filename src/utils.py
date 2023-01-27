def detach_lstm(states):
    return [state.detach() for state in states]

def detach_rnn(hidden):
    return hidden.detach()

def print_msg(msg):
    msg = "## {} ##".format(msg)
    length = len(msg) 
    msg = "\n{}\n".format(msg)
    print(length*"#" + msg + length * "#")

