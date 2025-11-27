from sys import argv

from simulator import RaceTrack, Simulator, plt

if __name__ == "__main__":
    assert(len(argv) == 3)
    racetrack = RaceTrack(argv[1])
    racelinePath = argv[2]
    simulator = Simulator(racetrack, racelinePath) # add racelinePath
    simulator.start()
    plt.show()