import numpy as np
import matplotlib.pyplot as plt 
from environment import *




def normalize(a):
    return a / a.sum()

def predict(belief,transition_matrix):
    shape = belief.shape
    if belief.ndim != 1:
        belief = belief.ravel()
    return np.reshape(belief.dot(transition_matrix),shape)

def effective_weights(weights):
    return 1. / np.sum(np.square(weights))

def get_particles_weights(prior,num_samples,size):
    rng = np.random.default_rng()
    particles = rng.choice(np.arange(size),num_samples, p=prior.flatten())
    weights = normalize(np.ones_like(particles))
    return particles, weights
    

def resample_particles(particles, weights):
    num_samples = len(particles)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1. 
    
    indexes = np.searchsorted(cumulative_sum, np.random.rand(num_samples))
    particles[:] = particles[indexes]
    weights.fill(1.0 / num_samples)
    return particles, weights 

def update_beliefs(observations,emission_matrix,belief):
    for obs in observations:
        likelihood = emission_matrix[obs.value]
        new_belief = likelihood[1] * belief[obs.pos]
        neg_belief = likelihood[0] * (1-belief[obs.pos])
        normalized_new_belief = new_belief / (new_belief + neg_belief)
        belief[obs.pos] = normalized_new_belief
    return normalize(belief)
        
        

def get_emission_matrix(alpha,beta):
    return np.array([[1-beta, 1-alpha], [beta,alpha]])


def sample_observations(state,emission_matrix,num_obs=3):
    grid_size = state.shape
    inds = np.random.randint(0,grid_size[0]*grid_size[1],size=3)
    rows,cols = np.unravel_index(inds,grid_size)
    sampled_states = [state[r,c] for r,c in zip(rows,cols)]
    #print(inds,sampled_states)
    emission_probs = emission_matrix[:,np.array(sampled_states,dtype=int)]
    #print(emission_probs)
    sampled_obs = np.random.rand(num_obs)
    #print(sampled_obs)
    maxes = emission_probs.max(axis=0)
    ind_max = emission_probs.argmax(axis=0)
    mask = sampled_obs < maxes
    obs = np.zeros((num_obs,))
    obs = []
    #print(maxes,ind_max,mask,obs)
    for i in range(len(mask)):
        val = ind_max[i] if mask[i] else 1 - ind_max[i]
        pos = (rows[i],cols[i])
        obs.append(Observation(pos,val))
    return obs

def sample_next_state(state,transition_matrix):
    sample_probs = transition_matrix[state.ravel()==1].squeeze()
    ind = np.random.choice(np.arange(sample_probs.size),p=sample_probs)
    row,col = np.unravel_index(ind,state.shape)
    new_state = np.zeros(state.shape)
    new_state[row,col] = 1
    return new_state

class HMM:
    def __init__(self,transition_matrix,emission_matrix,prior,state):
        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix
        self.prior = prior
        self.state = state
    

    @classmethod
    def basic_initialization(cls,state,alpha=0.9,beta=0.2):
        grid_size = state.shape
        transition_matrix = np.eye(grid_size[0]**2,grid_size[1]**2)
        emission_matrix = get_emission_matrix(alpha,beta)
        prior = np.random.uniform(size=grid_size)
        prior = normalize(prior)
        return cls(transition_matrix,emission_matrix,prior,state)

    def particle_predict(self,particles):
        # if isinstance(particles,list):
        #     particles = np.stack(particles)
        #particle_probs = particles.dot(transition_matrix)
        particle_probs = self.transition_matrix[particles]
        rng = np.random.default_rng()
        sample_particles = np.array([rng.choice(np.arange(particle_probs.shape[1]),p=particle_probs[i]) for i in range(len(particle_probs))])
        return sample_particles

    def update_weights(self,particles,weights,observations):
        for o in observations:
            ind = o.get_1dpos(self.state.shape)
            likelihood = self.emission_matrix[o.value][1]
            particle_indices = np.where(particles == ind)
            weights[particle_indices] *= likelihood
        normalize(weights)
        return weights

    def get_estimated_belief(self,pos_to_prob):
        estimated_belief = np.zeros_like(self.state)
        for key,val in pos_to_prob.items():
            row,col = np.unravel_index(key,self.state.shape)
            estimated_belief[row,col] = val
        return estimated_belief

    def estimate_state(self,particles,weights):
        positions,counts = np.unique(particles,return_counts=True) 
        probs = weights
        pos_to_prob = {}
        for i in range(len(particles)):
            particle_pos = particles[i]
            if particles[i] in pos_to_prob:
                pos_to_prob[particle_pos] += probs[i]
            else:
                pos_to_prob[particle_pos] = probs[i]
        return pos_to_prob

    def run(self,length):
        transition_belief_history = [self.prior]
        correction_belief_history = [self.prior]
        state_history = [self.state]
        obs_history = []
        # In the case that we know the prior
        belief = self.prior
        for t in range(1,length):
            new_state = sample_next_state(state_history[t-1],self.transition_matrix)
            state_history.append(new_state)
            observations = sample_observations(new_state,self.emission_matrix)
            obs_history.append(observations)
            belief = predict(belief,self.transition_matrix)
            #print("Transition Model Belief: ", belief)
            transition_belief_history.append(belief)
            belief =update_beliefs(observations,self.emission_matrix,belief)
            #print("Corrected Belief: ", belief)
            correction_belief_history.append(belief)
        return correction_belief_history

    def particle_filter(self,num_samples,length,threshold):
        num_states = self.state.shape[0]*self.state.shape[1]
        particles,weights = get_particles_weights(self.prior,num_samples,num_states)
        state_history = [self.state]
        particle_history = [particles]
        weight_history = [weights]
        estimated_belief = self.get_estimated_belief(self.estimate_state(particles,weights))
        estimated_belief_history = [estimated_belief]

        for t in range(1,length):
            # Get next state
            new_state = sample_next_state(state_history[t-1],self.transition_matrix)
            state_history.append(new_state)

            # Get observations
            observations = sample_observations(new_state,self.emission_matrix)

            # Predict next state
            particles = self.particle_predict(particles)
            particle_history.append(particles)
            # Update particle weights
            weights = self.update_weights(particles,weights,observations)
            weight_history.append(weights)

            # Resample if needed
            if effective_weights(weights) < threshold:
                particles,weights = resample_particles(particles,weights)

            pos_to_prob = self.estimate_state(particles,weights)
            estimated_belief = self.get_estimated_belief(pos_to_prob)
            estimated_belief_history.append(estimated_belief)
        
        return estimated_belief_history,particle_history,weight_history,state_history

    def plot_particles(self,state,particles,estimated_belief,alpha=0.20):
        fig = plt.figure()
        if particles.ndim == 1:
            rows,cols = np.unravel_index(particles,self.state.shape)
        else:
            rows = pos_particles[:,0]
            cols = pos_particles[:,1]
        plt.scatter(rows,cols,color='g',alpha=alpha)
        actual_pos = np.where(state == 1)
        actual_state = plt.scatter(actual_pos[0][0],actual_pos[1][0],marker='+',color='k', s=180, lw=3)
        row_index,col_index = np.unravel_index(estimated_belief.argmax(),self.state.shape)
        estimated_state = plt.scatter(row_index,col_index,marker='s', color='r')
        plt.legend([actual_state,estimated_state],['Actual Position','Estimated Position'],loc=4, numpoints=1)
        plt.xlim(0,estimated_belief.shape[0])
        plt.ylim(0,estimated_belief.shape[1])
        plt.show()

    def plot_belief(self,belief,save_fname=''):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x_data, y_data = np.meshgrid(np.arange(belief.shape[1]),np.arange(belief.shape[0]))
        x_data = x_data.flatten()
        y_data = y_data.flatten()
        z_data = belief.flatten()
        bar = ax.bar3d(x_data,
                 y_data,
                 np.zeros(len(z_data)),
                        1, 1, z_data, alpha=0.5)
        
        # anim = animation.FuncAnimation(fig,update,len(beliefs),fargs=(beliefs,bar), interval=1000, blit=False)
        # anim.save('mymovie.mp4',writer=animation.FFMpegWriter(fps=10))
        if save_fname:
            plt.savefig(save_fname)
        plt.show()


    def plot_beliefs(self,beliefs):
        def update(i,beliefs,bar):
            bar.set_3d_properties([beliefs[i].flatten()])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x_data, y_data = np.meshgrid(np.arange(beliefs[0].shape[1]),np.arange(beliefs[0].shape[0]))
        x_data = x_data.flatten()
        y_data = y_data.flatten()
        z_data = beliefs[0].flatten()
        bar = ax.bar3d(x_data,
                 y_data,
                 np.zeros(len(z_data)),
                        1, 1, z_data, alpha=0.5)
        
        # anim = animation.FuncAnimation(fig,update,len(beliefs),fargs=(beliefs,bar), interval=1000, blit=False)
        # anim.save('mymovie.mp4',writer=animation.FFMpegWriter(fps=10))
        plt.show()



if __name__ == "__main__":
    state = np.zeros((20,20))
    state[5,5] = 1
    hmm = HMM.basic_initialization(state)
    samples = 2000
    estimated_belief_history,particle_history,weight_history,state_history = hmm.particle_filter(samples,400,samples/2)
    #hmm.plot_particles(state_history[1],particle_history[1],estimated_belief_history[1])
    #hmm.plot_particles(state_history[-1],particle_history[-1],estimated_belief_history[-1])
    # beliefs = hmm.run(400)
    # print(beliefs[-1])
    hmm.plot_belief(estimated_belief_history[-1],"pfilterbeliefend.png")
