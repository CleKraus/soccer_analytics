import math
import copy
import numpy as np
import pandas as pd


class Player(object):
    def __init__(self, row):

        # time between two frames
        self.dt = 0.04
        self.id = row["playerId"]
        self.teamname = row["team"]
        self.get_position(row)
        self.get_velocity(row)
        # pass interception probability
        self.PIP = 0.0

        # initialize t_int which stores the first potential interception point of a player with
        # a played ball
        self.t_int = None

        # maximal speed of a player
        self.V_max = 7.8

        # reachable height of a player in meters
        self.player_height = 2

        # parameters needed for computation of the reachable points of a player as outlined by Peralta et al
        self.alpha = 1.3
        self.sigma = 0.45

    def get_position(self, row):
        """
        Return the position of the player
        """
        self.position = np.array([row["xPos"], row["yPos"]])
        self.inframe = not np.any(np.isnan(self.position))

    def get_velocity(self, row):
        """
        Return the velocity of the player
        """
        self.velocity = np.array([row["vx"], row["vy"]])
        if np.any(np.isnan(self.velocity)):
            self.velocity = np.array([0.0, 0.0])

    def compute_center_and_radius(self, t):
        """
        Compute the reachable circle of a player after *t* frames according the formula outlined by Peralta et al
        in their paper. Notice that this takes into account the current velocity of the player
        """

        # center position after *t* frames
        center = (
            self.position
            + (1 - math.exp(-self.alpha * t * self.dt)) / self.alpha * self.velocity
        )

        # radius after *t* frames
        radius = self.V_max * (
            t * self.dt - (1 - math.exp(-self.alpha * t * self.dt)) / self.alpha
        )

        return center, radius

    def interception_probability(self, t, ball_pos):
        """
        Given the ball position after *t* frames, the function computes the probability that the player has
        intercepted the ball after *t* frames.
        """

        # compute the reachable circle by the player
        center, radius = self.compute_center_and_radius(t)

        # in case the ball was played on the ground
        if len(ball_pos) == 2:
            reach = np.linalg.norm(center - ball_pos) < radius

        # if the pass was a high pass, we also check that of the ball is at most *player_height* in order
        # to be reachable
        if len(ball_pos) == 3:
            reach = (np.linalg.norm(center - ball_pos[:2]) < radius) and (
                ball_pos[2] < self.player_height
            )

        # if the player is potentially able to reach the ball
        if reach:
            # if this is the first time the player potentially reaches the ball
            if self.t_int is None:
                self.t_int = t

            # compute the probability to intercept
            P_int = 1 / (
                1.0
                + np.exp(
                    -np.pi / np.sqrt(3.0) / self.sigma * (t - self.t_int) * self.dt
                )
            )

        # if the player is definitely not able to reach the ball
        else:

            # if he is not able to reach the ball any more but was able to reach it before (i.e. the ball
            # bypassed him), we set the earlist interception time to 0
            if self.t_int is not None:
                self.t_int = None

            P_int = 0

        return P_int


def set_team_players(df_frame, team):
    """
    Set up the team with all players for a given frame
    """
    team_players = []
    for _, row in df_frame.iterrows():
        if row["team"] == team:
            team_players.append(Player(row))
    return team_players


class PassProbabilityModeler:
    """
    Computation of the pass success probability using a model proposed by Peralta et al. ("Seeing in to the future:
    using self-propelled particle models to aid player decision-making in soccer") and Spearman et al. ("Physics-Based
    Modeling of Pass Probabilities in Soccer")

    Main functions to use are:
        - set_current_frame to initialize the model with a certain frame
        - get_pass_success_probability calculating the success probability for a given pass

    """

    def __init__(self, three_dim=True):

        # time difference between two frames
        self.dt = 0.04
        # players need to intercept pass within 250 frames (i.e. 10 seconds)
        self.max_frames = 250

        # player params as used in papers
        self.lambda_player = 4.3
        self.player_height = 2

        self.eps = 0.00001

        # ball params as used in papers
        self.m = 0.42
        self.rho = 1.225
        self.drag = 0.25
        self.area = 0.038
        self.dt = 0.04
        self.mu = 0.55
        self.g = 9.81
        self.ball_position = [0, 0]

        # ball factor to compute ball trajectory for ground passes
        self.ball_factor = -1 / self.m * self.rho * self.drag * self.area

        # times when ground passes stop for different initial speeds
        self.ball_stop_times = self._get_stopping_time_by_resistance()

        # computation of the minimum required angle for the ball to reach 2m of height
        if three_dim:
            self.min_angles_overpass = self._get_minimum_required_angle_to_overpass()

        # frame variables
        self.defending_players = None
        self.attacking_players = None

        # pass variables
        self.ball_pos_per_frame = None

    def set_current_frame(self, df_frame, team_ball_position):
        """
        Given a frame of the match, set up the player classes as well as the ball position

        :param df_frame: (pd.DataFrame) Data frame containing one frame of the match
        :param team_ball_position: (str) Team that is currently in position of the ball ("Home" or "Away")
        """

        self.frame = df_frame.copy()

        def_team = "Home" if team_ball_position == "Away" else "Away"
        att_team = "Away" if team_ball_position == "Away" else "Home"

        self.defending_players = set_team_players(df_frame, def_team)
        self.attacking_players = set_team_players(df_frame, att_team)

        df_ball = df_frame[df_frame["playerId"] == -1].iloc[0]
        self.ball_position = np.array([df_ball["xPos"], df_ball["yPos"]])

    def compute_ball_position_at_every_time(self, angle, speed):
        """
        Computation of the ball trajectory given an *angle* and initial *speed*. Notice that both 2-dim and 3-dim
        trajectories can be computed. For ground passes, the approach suggested by Peralta et al. is used while for
        air passes, the approach suggested by Spearman et al. is used.

        """
        tmp_r = copy.deepcopy(self.ball_position)

        # in case the angle is a vector, i.e. air pass
        if type(angle) != float and type(angle) != int:

            # if the polar angle is very small, we assume a ground pass
            if angle[1] < self.eps:
                angle = angle[0]

            else:
                angle[1] = np.pi / 2 - angle[1]

                # compute the pass direction
                pass_direction = np.array(
                    [
                        math.cos(angle[0]) * math.sin(angle[1]),
                        math.sin(angle[0]) * math.sin(angle[1]),
                        math.cos(angle[1]),
                    ]
                )

                # every pass is assumed to start on the ground
                tmp_r = np.array(list(tmp_r) + [0.0])

        # in case of a ground pass
        if type(angle) == float or type(angle) == int:
            pass_direction = np.array([math.cos(angle), math.sin(angle)])

        tmp_v = speed * pass_direction

        # set the initial position of the ball
        r = [tmp_r]

        # model in case of a ground pass
        if type(angle) == float or type(angle) == int:

            t_max = self.ball_stop_times[speed]

            for i in np.arange(t_max):

                # Peralta et al suggest that the first 2/3 of the pass trajectory was mainly caused by aerodynamics,
                # while the last third was caused by grass friction
                if i < 2 / 3 * t_max:
                    tmp_a = self.ball_factor * np.linalg.norm(tmp_v) * tmp_v
                else:
                    tmp_a = -1 * self.mu * self.g * pass_direction

                tmp_v = tmp_v + tmp_a * self.dt
                tmp_r = tmp_r + tmp_v * self.dt
                r.append(tmp_r)

        # model in case of an air pass
        else:

            for _ in np.arange(self.max_frames - 1):
                a_direction = self.ball_factor * np.linalg.norm(tmp_v[:2]) * tmp_v[:2]
                a_height = -1 * self.g

                tmp_a = np.array(list(a_direction) + [a_height])

                tmp_v = tmp_v + tmp_a * self.dt
                tmp_r = tmp_r + tmp_v * self.dt
                tmp_r[2] = np.max([tmp_r[2], 0.0])
                r.append(tmp_r)

        return r

    def _ball_stopped_by_resistance(self, speed, T):
        """
        Helper function to determine whether the ball has stopped before T frames due to friction of the grass or not
        when using the model as proposed by Peralta
        """

        tmp_r = np.array([0.0, 0.0])
        pass_direction = np.array([1, 0])

        tmp_v = speed * pass_direction

        for i in np.arange(T):

            if i < 2 / 3 * T:
                tmp_a = self.ball_factor * np.linalg.norm(tmp_v) * tmp_v
            else:
                tmp_a = -1 * self.mu * self.g * pass_direction

            if any(np.abs(tmp_v) - np.abs(tmp_a * self.dt) < -1 * self.eps):
                return True

            tmp_v = tmp_v + tmp_a * self.dt
            tmp_r = tmp_r + tmp_v * self.dt

        return False

    def _get_stopping_time_by_resistance(self, min_speed=1, max_speed=20):
        """
        Helper function to determine after how many frames a ground pass with different initial speeds stops, when
        using the model as proposed by Peralta
        """

        dict_t_max = {}

        for speed in np.arange(min_speed, max_speed + 1):

            min_val = 0
            max_val = 300
            mid_val = int((max_val + min_val) / 2)

            while True:
                stopped_ball = self._ball_stopped_by_resistance(speed, mid_val)
                if stopped_ball:
                    max_val = mid_val
                    mid_val = int((max_val + min_val) / 2)
                else:
                    min_val = mid_val
                    mid_val = int((max_val + min_val) / 2)

                if min_val == mid_val:
                    break

            dict_t_max[speed] = mid_val

        return dict_t_max

    def _get_minimum_required_angle_to_overpass(self, min_speed=1, max_speed=20):
        """
        Helper function to compute for each initial speed the minimum angle to overpass a player
        """

        dict_z_angle = dict()

        # loop through all potential speeds
        for speed in np.arange(min_speed, max_speed + 1):

            # check if already with pi/3 the ball never reaches a height > player height
            r = self.compute_ball_position_at_every_time([0, np.pi / 3], speed)

            if max([pos[2] for pos in r]) < self.player_height:
                min_angle = np.pi / 3
            else:
                # loop through different angles and search for the first angle where the ball reaches player height
                for angle in np.arange(np.pi / (3 * 30), np.pi / 3, np.pi / (3 * 30)):
                    r = self.compute_ball_position_at_every_time([0, angle], speed)

                    if max([pos[2] for pos in r]) > self.player_height:
                        min_angle = angle - np.pi / (3 * 30)
                        break

            dict_z_angle[speed] = min_angle

        return dict_z_angle

    def get_pass_success_probability(self, angle, speed, specific_players=None):
        """
        Given the angle and speed of a pass, function returns the probability that the pass is successful, i.e.
        stays within the attacking team. Furthermore, a data frame with all ball positions, for which an interception
        is possible is returned together with the probability of the interception happening at exactly that position.
        If *angle* is two-dimensional, an air pass is assumed, while when angle only is one value, a ground pass is
        assumed.

        :param angle: (float or list of floats) Angle of the pass (float if ground pass, list of floats when air pass)
        :param speed: (float) Initial speed of the pass
        :param specific_players: (int or list) If not None, it is assumed that only the *specific_players* try to
                                 reach the ball
        :return: float with the probability of the pass being successful and pd.DataFrame with the possible interception
                points of the pass
        """

        # specific players of attacking team only

        # make sure specific players is always a list
        if type(specific_players) == int:
            specific_players = [specific_players]

        # pass probability interceptions are set to 0
        for player in self.attacking_players:
            player.PIP = 0

        for player in self.defending_players:
            player.PIP = 0

        # compute ball positions at the different times
        r = self.compute_ball_position_at_every_time(angle, speed)

        # pass probability interceptions of attacking and defending team are set to 0
        pass_prob_att = np.zeros(self.max_frames)
        pass_prob_def = np.zeros(self.max_frames)

        # make sure that the ball position vector also has length *self.max_frames*
        r = np.array(r + [r[-1]] * (self.max_frames - len(r)))

        # initialize variables
        ball_in_play = True
        total_proba = 0
        t = 1

        # loop over frames as long as ball is in play and the total proability of interception
        # is not yet ~ 1
        while 1 - total_proba > self.eps and ball_in_play and t < self.max_frames:

            # ball position in frame *t*
            ball_pos = r[t]

            # loop through all players of the attacking team
            for player in self.attacking_players:

                # if only specific players are considered
                if specific_players is not None and player.id not in specific_players:
                    continue

                # compute the player's passing interception probability within *t* frames
                P_int = player.interception_probability(t, ball_pos)
                dPdT = (
                    (1 - pass_prob_att[t - 1] - pass_prob_def[t - 1])
                    * P_int
                    * self.lambda_player
                )

                player.PIP += dPdT * self.dt

                # add this to the team's passing interception probability
                pass_prob_att[t] += player.PIP

            # loop through all defending players and do the same thing
            for player in self.defending_players:
                P_int = player.interception_probability(t, ball_pos)
                dPdT = (
                    (1 - pass_prob_att[t - 1] - pass_prob_def[t - 1])
                    * P_int
                    * self.lambda_player
                )
                player.PIP += dPdT * self.dt
                pass_prob_def[t] += player.PIP

            # check whether the ball is still in play. If not, the defending team gets all the
            # remaining probability
            if (
                (ball_pos[0] > 105)
                or (ball_pos[0] < 0)
                or (ball_pos[1] > 68)
                or (ball_pos[1] < 0)
            ):
                ball_in_play = False
                pass_prob_def[t] = 1 - pass_prob_att[t - 1]
                pass_prob_att[t] = pass_prob_att[t - 1]

            # get the total probability of the pass being intercepted after at most *t* frames
            total_proba = pass_prob_att[t] + pass_prob_def[t]

            t += 1

        # get all frames with a positive probability of intercepting the ball
        poss_intercept = pass_prob_att + pass_prob_def > self.eps
        rel_a = pass_prob_att[poss_intercept]
        rel_d = pass_prob_def[poss_intercept]
        rel_r = r[poss_intercept]
        rel_t = np.arange(self.max_frames)[poss_intercept] * self.dt

        # make sure the total interception probability is at most 1
        last_prob = rel_d[-1] + rel_a[-1]
        rel_d[-1] /= last_prob
        rel_a[-1] /= last_prob

        # get the probability for the ball to be touched in every frame
        prob_a = np.insert(np.diff(rel_a), 0, rel_a[0])
        prob_d = np.insert(np.diff(rel_d), 0, rel_d[0])

        # save all the relevant information in a data frame
        df_interception = pd.DataFrame()
        df_interception["defProba"] = prob_d
        df_interception["attProba"] = prob_a
        df_interception["time"] = rel_t
        df_interception["xPos"] = rel_r[:, 0]
        df_interception["yPos"] = rel_r[:, 1]

        # in case we are looking at a high pass, also save the height angle
        if type(angle) != float and type(angle) != int:
            df_interception["zPos"] = rel_r[:, 2]
            df_interception["angle"] = angle[0]
            df_interception["angleHeight"] = angle[1]
        else:
            df_interception["angle"] = angle

        df_interception["speed"] = speed

        return rel_a[-1], df_interception
