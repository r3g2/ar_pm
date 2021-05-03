from environment import *
from hmm import *

def test_update_beliefs():

    observations = [Observation((i,i),i%2) for i in range(3)]
    prior = np.zeros((3,3))
    prior[0,0] = 1
    prior[1,1] = 0
    B = get_emission_matrix(0.9,0.2)
    new_beliefs = update_beliefs(observations,B,prior)
    print(new_beliefs)


def test_run_plotting():

    state = np.zeros((20,20))
    state[5,5] = 1
    hmm = HMM.basic_initialization(state)
    beliefs = hmm.run(400)
    print(beliefs[-1])
    hmm.plot_belief(beliefs[-1],"beliefpropend.png")


def test_estimate():
    hmm = HMM.basic_initialization(np.zeros((7,7)))
    particles = np.array([3,0,3,2])
    weights = np.array([0.2,0.4,0.3,0.1])
    print(hmm.estimate_state(particles,weights))


def test_resample_particles():
    particles = np.array([3,0,3,2])
    weights = np.array([0.2,0.4,0.3,0.1])
    print(resample_particles(particles,weights))

if __name__ == "__main__":
    test_resample_particles()