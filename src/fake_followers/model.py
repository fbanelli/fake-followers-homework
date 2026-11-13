'''Agents of the platform model: seekers (users), content creators (CCs), the
follower network, the recommender system and the platform that runs the rounds.

Creators are indexed by quality from 0 (best).
'''

import numpy as np


class User:
    def __init__(self, config, id):
        self.config = config
        self.id = id

        # the best CC followed so far
        self.best_followed_cc = None

    def decide_follow(self, c):
        '''Follows CC c iff it is better (closer to the top) than the best followed so far.'''

        if (self.best_followed_cc is None) or (self.best_followed_cc.id > c.id):
            self.best_followed_cc = c
            return True

        return False


class CC:
    def __init__(self, config, id):
        self.config = config
        self.id = id


class Network:
    '''Follower counts of the content creators.'''

    def __init__(self, config):
        self.config = config
        num_ccs = config['num_ccs']

        # real followers: seekers who chose to follow the creator
        self.num_followers = np.zeros(num_ccs, dtype=int)

        # Fake followers bought by strategic creators (set by Platform). Fake accounts
        # follow only their buyer and never follow recommendations, so they are a fixed
        # per-creator count.
        self.fake_followers = np.zeros(num_ccs, dtype=int)

    @property
    def total_followers(self):
        '''Follower counts as the platform sees them (real + fake).
        This is what the recommender uses, and what ranks creators for payoffs.'''
        return self.num_followers + self.fake_followers

    def follow(self, u, c):
        '''User u is shown content creator c and follows it if it is better than
        everyone u follows. A seeker never follows the same creator twice, since a
        creator it already follows is never better than its best followed one.'''

        if u.decide_follow(c):
            self.num_followers[c.id] += 1


class RS:
    '''The recommender system: shows one creator to each seeker in every round.

    rs_model 'general': creator i is shown with probability proportional to
        (1 + followers of i) ** alpha (alpha = -1 AntiPA, 0 UR, 1 PA).
    rs_model 'ExtremePA': one of the most followed creators, uniformly at random.
    '''

    def __init__(self, config, rng):
        '''rng - the run's random generator (np.random.RandomState); all randomness
        in the recommendations is drawn from it, so a run is reproducible from its seed.'''
        self.config = config
        self.rng = rng

    def recommend_general(self, content_creators, num_followers):
        num_users = self.config['num_users']
        num_ccs = self.config['num_ccs']
        alpha = self.config['alpha']

        prob_choice = (num_followers + np.ones(num_ccs))**alpha
        prob_choice /= sum(prob_choice)

        return self.rng.choice(content_creators, num_users, p=prob_choice)

    def most_popular(self, content_creators, num_followers):
        '''The CCs that have the maximum number of followers.'''

        top = max(num_followers)
        return [c for c in content_creators if num_followers[c.id] == top]

    def recommend_extreme_pa(self, content_creators, num_followers):
        num_users = self.config['num_users']
        return self.rng.choice(self.most_popular(content_creators, num_followers), num_users)

    def recommend(self, content_creators, num_followers):
        '''A list of recommendations, one per user.'''

        if self.config['rs_model'] == 'general':
            return self.recommend_general(content_creators, num_followers)
        if self.config['rs_model'] == 'ExtremePA':
            return self.recommend_extreme_pa(content_creators, num_followers)
        raise ValueError(f"Unknown rs_model {self.config['rs_model']!r}; use 'general' or 'ExtremePA'")


class Platform:
    def __init__(self, config):
        self.config = config

        # the platform keeps track of the number of timesteps it has been iterated
        self.timestep = 0

        self.network = Network(config)
        self.users = [User(config, i)
                      for i in range(config['num_users'])]
        self.ccs = [CC(config, i)
                    for i in range(config['num_ccs'])]
        # one random generator per run, seeded from the config
        self.rng = np.random.RandomState(config['random_seed'])
        self.rs = RS(config, self.rng)

        # strategic creators: each creator in buying_creators adds its own budget of fake followers at t=0
        self.buy_fake_followers()

        # average (0-based) position of the best creator each seeker follows, after every round
        self.average_pos_best_cc = []

        # the users who can still find a better creator
        self.id_searching_users = list(range(self.config['num_users']))

    def buy_fake_followers(self):
        '''Adds the purchased fake followers before the first round.

        Config keys (both optional; without them nobody buys):
            budgets         - {creator id: budget}: the creators who can buy (the wealthy ones,
                              0 = best quality) and how many fake followers each one can buy
            buying_creators - the strategy profile: ids of wealthy creators who buy
        '''

        budgets = self.config.get('budgets') or {}
        for c in self.config.get('buying_creators', []):
            if c not in budgets:
                raise ValueError(f'creator {c} buys followers but has no entry in budgets')
            self.network.fake_followers[c] += budgets[c]

    def iterate(self):
        '''Runs one round: every seeker gets a recommendation and decides whether to follow it.'''

        self.timestep += 1

        recs = self.rs.recommend(self.ccs, self.network.total_followers)
        for u in self.users:
            self.network.follow(u, recs[u.id])

        # record the average position of the best creator followed by seekers
        average_pos = 0
        num_users = self.config['num_users']
        for u in self.users:
            average_pos += u.best_followed_cc.id / num_users
        self.average_pos_best_cc.append(average_pos)

    def update_searching_users(self):
        '''Updates the list of users who can still find a better CC than their best one.'''

        if self.config['rs_model'] == 'ExtremePA':
            # only the most followed creators can be recommended, so a seeker is still
            # searching only if one of them is better than its best followed creator
            most_popular_ccs = self.rs.most_popular(self.ccs, self.network.total_followers)
            best_reachable = min(c.id for c in most_popular_ccs)
            self.id_searching_users = [i for i in self.id_searching_users
                                       if best_reachable < self.users[i].best_followed_cc.id]
        else:
            # every creator can be recommended, so seekers search until they follow the best
            self.id_searching_users = [i for i in self.id_searching_users
                                       if self.users[i].best_followed_cc.id != 0]

    def check_convergence(self):
        '''The platform converged if no user can find a better CC any more.'''
        return len(self.id_searching_users) == 0
