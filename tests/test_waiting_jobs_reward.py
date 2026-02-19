import unittest

from jssp_core.reward_functions import WaitingJobsReward
from jssp_core.schedule import Schedule


class TestWaitingJobsReward(unittest.TestCase):
    def test_waiting_jobs_reward(self):
        # Create a simple instance: 2 jobs, 2 machines
        # Job 0: (0, 10), (1, 10)
        # Job 1: (1, 10), (0, 10)
        instance = [[(0, 10), (1, 10)], [(1, 10), (0, 10)]]
        schedule = Schedule(instance)

        reward_fn = WaitingJobsReward(schedule)

        # Initial state: both jobs are eligible (waiting)
        # eligible jobs: [0, 1]
        reward_data = {
            "state_before": schedule,
            "state_after": None,  # Not used
            "is_complete": False,
        }

        reward = reward_fn.calculate_reward(reward_data)
        self.assertEqual(reward, -2.0)

        # Schedule job 0, op 0
        schedule.schedule_job(0)

        # Now job 0 is waiting for op 1 (because op 0 is scheduled)
        # Job 1 is waiting for op 0
        # eligible jobs: [0, 1]
        # Wait, if we schedule job 0 op 0, job 0 op 1 becomes eligible immediately?
        # Let's check Schedule logic.
        # Yes:
        # self.eligible_operations[(job_id, op_id)] = 0
        # if op_id + 1 < len(self.instance[job_id]):
        #    self.eligible_operations[(job_id, op_id + 1)] = 1

        reward = reward_fn.calculate_reward(reward_data)
        self.assertEqual(reward, -2.0)

        # Schedule job 1, op 0
        schedule.schedule_job(1)
        reward = reward_fn.calculate_reward(reward_data)
        self.assertEqual(reward, -2.0)

        # Schedule job 0, op 1 (last op)
        schedule.schedule_job(0)
        # Job 0 is now finished.
        # eligible jobs: [1]
        reward = reward_fn.calculate_reward(reward_data)
        self.assertEqual(reward, -1.0)

        # Schedule job 1, op 1 (last op)
        schedule.schedule_job(1)
        # Job 1 is now finished.
        # eligible jobs: []
        reward = reward_fn.calculate_reward(reward_data)
        self.assertEqual(reward, -0.0)


if __name__ == "__main__":
    unittest.main()
