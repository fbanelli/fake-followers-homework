'''Runs one simulation of the platform.'''

from . import model


def simulate(config):
    '''Runs the platform for config['num_steps'] rounds and returns, for every round in
    config['record_at_timesteps']:

        num_followers   - the real follower count of every creator
        dissatisfaction - the quality rank (1 = best) of the best creator each seeker
                          follows, averaged over seekers

    and the fake followers of every creator (constant over the run). Once no seeker can
    find a better creator any more, nothing changes, so the remaining rounds are skipped.
    '''

    p = model.Platform(config)
    record_at_timesteps = config['record_at_timesteps']
    results = {'num_followers': [], 'dissatisfaction': []}

    did_converge = False
    for i in range(1, config['num_steps'] + 1):
        if not did_converge:
            p.iterate()

        if i in record_at_timesteps:
            if not did_converge:
                p.update_searching_users()
                did_converge = p.check_convergence()
            results['num_followers'].append(p.network.num_followers.tolist())
            # +1 turns the 0-based id of the best followed creator into a 1-based rank
            results['dissatisfaction'].append(p.average_pos_best_cc[-1] + 1)

    results['fake_followers'] = p.network.fake_followers.tolist()
    return results
