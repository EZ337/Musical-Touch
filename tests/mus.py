from scamp import *

def playChord(chord : list[int], vol : int, length : int):
    piano.play_chord(chord, vol, length)


s = Session()

piano = s.new_part("piano")

text = "a b c"
 
CL = {'c' : [60, 64, 67], 
    'f' : [65, 69, 72], 
    'g' : [67, 71, 74],
    'am' : [69, 72, 74],
    'em' : [52, 55, 57],
    'dm' : [50, 53, 55]}

do_fa_so = [CL['c'], CL['f'], CL['g']]

do_fa_re_so = [CL['c'], CL['f'], CL['dm'], CL['g']]

do_so_la_fa = [CL['c'], CL['g'], CL['am'], CL['f']]


# piano.play_note(60, 2, 1)
# piano.play_note(65, 2, 1)
# piano.play_note(67, 2, 1)


wait(1, "time")
measures = 0
while (True):
    for chord in do_fa_re_so:
        measures += 1
        playChord(chord, 1, 1)

    # if (measures == 10):
    #     measures = 0
    
    # for chord in do_so_la_fa:
    #     measures += 1
    #     playChord(chord, 1, 1)

    # if (measures == 10):
    #     break


        
    


# for char in text:
#     if char == " ":
#         wait(0.2)
#     elif char.isalnum():
#         # piano.play_note(ord(char) - 20, 0.5, 0.06)
#         piano.play_chord([50, 80], 1, 1)
