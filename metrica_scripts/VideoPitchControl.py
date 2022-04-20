import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage
from tqdm import tqdm
from joblib import Parallel, delayed
import glob
import os
import os.path
import re
import matplotlib.animation as animation

sys.path.append('./LaurieOnTracking')
from metrica_scripts import Metrica_Viz as mviz
from metrica_scripts import Metrica_IO as mio
from metrica_scripts import Metrica_Velocities as mvel
from metrica_scripts import Metrica_PitchControl as mpc

# sys.path.append('./lastrow_to_fot/lastrow_to_friendsoftracking')
# import lastrow_to_friendsoftracking as lrfot

# from pitch_control_helpers import *

def initialise_persistent_play_data(data_attack, data_defence, play, field_dimen=(106.,68.), n_grid_cells_x=50):
   n_grid_cells_y = int(n_grid_cells_x*field_dimen[1]/field_dimen[0])
   xgrid = np.linspace( -field_dimen[0]/2., field_dimen[0]/2., n_grid_cells_x)
   ygrid = np.linspace( -field_dimen[1]/2., field_dimen[1]/2., n_grid_cells_y )
   
   data_attack_play = mvel.calc_player_velocities(data_attack.loc[(play)],smoothing=False)
   data_defence_play = mvel.calc_player_velocities(data_defence.loc[(play)],smoothing=False)
   
   return(data_attack_play, data_defence_play, xgrid, ygrid)


def generate_lvp_pc_time(data_attack_play, data_defence_play, xgrid, ygrid, t=0, frame=None, output_mode='plot'):
   fps = 20
   if not frame: frame = int(fps*t)

   params=mpc.default_model_params(2)

   ball_pos = np.array([data_attack_play.loc[frame].ball_x, data_attack_play.loc[frame].ball_y])


   PPCFa = np.zeros( shape = (len(ygrid), len(xgrid)) )
   PPCFd = np.zeros( shape = (len(ygrid), len(xgrid)) )

   attacking_players = mpc.initialise_players(data_attack_play.loc[frame],'attack', params=params)
   defending_players = mpc.initialise_players(data_defence_play.loc[frame],'defense', params=params)

   for i in range( len(ygrid) ):
         for j in range( len(xgrid) ):
               target_position = np.array( [xgrid[j], ygrid[i]] )
               PPCFa[i,j],PPCFd[i,j] = mpc.calculate_pitch_control_at_target(target_position, attacking_players, defending_players, ball_pos, params)
      # check probability sums within convergence
   
   # just return the raw output
   if output_mode=='raw':
      return(PPCFa)
   
   # returning some kind of visuals
   else:
      fig, ax = mviz.plot_pitchcontrol_for_liverpool(data_attack_play, data_defence_play, xgrid, ygrid, frame, PPCFd, annotate=True )

      if output_mode=='plot': 
         return(fig, ax)
      elif output_mode=='vid_artist':
         image = mplfig_to_npimage(fig)
         plt.close()
         return(image)
      else:
         print('Invalid output mode!')
   
def generate_full_lvp_pc(persistent_play_data, fps=20):
   full_play_pc = []
   for frame_num in tqdm(range(persistent_play_data[0].index.max()+1)):
      frame_pc = generate_lvp_pc_time(*(persistent_play_data), frame=frame_num, output_mode='raw') # tuple
      full_play_pc.append(frame_pc)
      
   return(full_play_pc)

def full_lvp_pc_video(persistent_play_data, full_play_pc, fps=20):
   n_frames = persistent_play_data[0].index.max() + fps
   t_length = n_frames/fps
   
   clip = mpy.VideoClip(lambda x: mviz.plot_pitchcontrol_for_liverpool(*(persistent_play_data), int(fps*x), full_play_pc[int(fps*x)], output_mode='vid_artist'), duration=t_length-1).set_fps(fps)
   return clip


def make_animation(hometeam,awayteam, events,fpath, fname='clip_test', figax=None, fps=25, 
                   team_colors=('r','b'), field_dimen = (106.0,68.0), 
                   include_player_velocities=False, PlayerMarkerSize=10, PlayerAlpha=0.7,dpi=100):
   length=len(hometeam)/fps

   clip = mpy.VideoClip(
      lambda x: draw_frame_x(hometeam, awayteam, events, t=x, figax=figax, team_colors=team_colors, field_dimen = field_dimen, include_player_velocities=include_player_velocities, PlayerMarkerSize=PlayerMarkerSize, PlayerAlpha=PlayerAlpha),
      duration=length).set_fps(fps)

   clip.write_videofile(fpath+fname+'.mp4')


def draw_frame_x(hometeam,awayteam, events t, figax=None, fps=25,team_colors=('r','b'), field_dimen = (106.0,68.0),include_player_velocities=False, PlayerMarkerSize=10, PlayerAlpha=0.7,dpi=100):
   params=parameters()
   fig,ax = plot_pitch_control_for_frame(int(t*fps), hometeam, awayteam, events, params)

   
   image = mplfig_to_npimage(fig)
   plt.close()
   return image

def save_match_clip(hometeam,awayteam, events, fpath, fname='clip_test', figax=None, frames_per_second=25, team_colors=('r','b'), field_dimen = (106.0,68.0), include_player_velocities=False, PlayerMarkerSize=10, PlayerAlpha=0.7, players_fixed=[],pith_control=False):
   """ save_match_clip( hometeam, awayteam, fpath )
   
   Generates a movie from Metrica tracking data, saving it in the 'fpath' directory with name 'fname'
   
   Parameters
   -----------
      hometeam: home team tracking data DataFrame. Movie will be created from all rows in the DataFrame
      awayteam: away team tracking data DataFrame. The indices *must* match those of the hometeam DataFrame
      fpath: directory to save the movie
      fname: movie filename. Default is 'clip_test.mp4'
      fig,ax: Can be used to pass in the (fig,ax) objects of a previously generated pitch. Set to (fig,ax) to use an existing figure, or None (the default) to generate a new pitch plot,
      frames_per_second: frames per second to assume when generating the movie. Default is 25.
      team_colors: Tuple containing the team colors of the home & away team. Default is 'r' (red, home team) and 'b' (blue away team)
      field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
      include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
      PlayerMarkerSize: size of the individual player marlers. Default is 10
      PlayerAlpha: alpha (transparency) of player markers. Defaault is 0.7
      players_fixed: players ti draw the line
      
   Returrns
   -----------
      fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
   """
   # check that indices match first
   print("a")
   assert np.all( hometeam.index==awayteam.index ), "Home and away team Dataframe indices must be the same"
   # in which case use home team index
   index = hometeam.index
   # Set figure and movie settings
   FFMpegWriter = animation.writers['ffmpeg']
   metadata = dict(title='Tracking Data', artist='Matplotlib', comment='Metrica tracking data clip')
   writer = FFMpegWriter(fps=frames_per_second, metadata=metadata)
   fname = fpath + '/' +  fname + '.mp4' # path and filename
   # create football pitch
   if figax is None:
      fig,ax = mviz.plot_pitch(field_dimen=field_dimen)
   else:
      fig,ax = figax
   fig.set_tight_layout(True)
   # Generate movie
   print("Generating movie...",end='')
   with writer.saving(fig, fname, 100):
      for i in index:
         params=parameters()
         plot_pitch_control_for_frame(i, hometeam, awayteam, events, params)
         # # TODO: params and goalkepers numbers
         # # get model parameters
         # params = mpc.default_model_params(3)
         # # find goalkeepers for offside calculation
         # GK_numbers = [mio.find_goalkeeper(hometeam),mio.find_goalkeeper(awayteam)]
         # PPCFa,xgrid,ygrid = generate_pitch_control_for_event(i,hometeam,awayteam,params,GK_numbers)
         # plot_pitchcontrol_for_event( i, events,  tracking_home, tracking_away, PPCF, annotate=True )
         # figobjs = [] # this is used to collect up all the axis objects so that they can be deleted after each iteration
         # for team,color in zip( [hometeam.loc[i],awayteam.loc[i]], team_colors) :
         #       x_columns = [c for c in team.keys() if c[-2:].lower()=='_x' and c!='ball_x'] # column header for player x positions
         #       y_columns = [c for c in team.keys() if c[-2:].lower()=='_y' and c!='ball_y'] # column header for player y positions
         #       objs, = ax.plot( team[x_columns], team[y_columns], color+'o', markersize=PlayerMarkerSize, alpha=PlayerAlpha ) # plot player positions
         #       figobjs.append(objs)
         #       if include_player_velocities:
         #          vx_columns = ['{}_vx'.format(c[:-2]) for c in x_columns] # column header for player x positions
         #          vy_columns = ['{}_vy'.format(c[:-2]) for c in y_columns] # column header for player y positions
         #          objs = ax.quiver( team[x_columns], team[y_columns], team[vx_columns], team[vy_columns], color=color, scale_units='inches', scale=10.,width=0.0015,headlength=5,headwidth=3,alpha=PlayerAlpha)
         #          figobjs.append(objs)
         # # plot ball
         # objs, = ax.plot( team['ball_x'], team['ball_y'], 'ko', markersize=6, alpha=1.0, linewidth=0)
         # figobjs.append(objs)
         # # include match time at the top
         # frame_minute =  int( team['Time [s]']/60. )
         # frame_second =  ( team['Time [s]']/60. - frame_minute ) * 60.
         # timestring = "%d:%1.2f" % ( frame_minute, frame_second  )
         # objs = ax.text(-2.5,field_dimen[1]/2.+1., timestring, fontsize=14 )
         # figobjs.append(objs)
         # # if there are players to observe
         # if players_fixed != []:
         #       for pid in players_fixed:
         #          objs, = ax.plot( hometeam['Home_'+str(pid)+'_x'], hometeam['Home_'+str(pid)+'_y'], 'r', markersize=6)
         #          figobjs.append(objs)
         # writer.grab_frame()
         # # Delete all axis objects (other than pitch lines) in preperation for next frame
         # for figobj in figobjs:
         #       figobj.remove()
   print("done")
   plt.clf()
   plt.close(fig)

def plot_pitchcontrol_for_event( frame, events,  tracking_home, tracking_away, PPCF, alpha = 0.7, include_player_velocities=True, annotate=False, field_dimen = (106.0,68)):
   """ plot_pitchcontrol_for_event( event_id, events,  tracking_home, tracking_away, PPCF )
   
   Plots the pitch control surface at the instant of the event given by the event_id. Player and ball positions are overlaid.
   
   Parameters
   -----------
      event_id: Index (not row) of the event that describes the instant at which the pitch control surface should be calculated
      events: Dataframe containing the event data
      tracking_home: (entire) tracking DataFrame for the Home team
      tracking_away: (entire) tracking DataFrame for the Away team
      PPCF: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team (as returned by the generate_pitch_control_for_event in Metrica_PitchControl)
      alpha: alpha (transparency) of player markers. Default is 0.7
      include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
      annotate: Boolean variable that determines with player jersey numbers are added to the plot (default is False)
      field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
      
   NB: this function no longer requires xgrid and ygrid as an input
      
   Returrns
   -----------
      fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
   """    

   # pick a pass at which to generate the pitch control surface
   pass_team = "Home"
   
   # plot frame and event
   fig,ax = mviz.plot_pitch(field_color='white', field_dimen = field_dimen)
   mviz.plot_frame( tracking_home.loc[frame], tracking_away.loc[frame], figax=(fig,ax), PlayerAlpha=alpha, include_player_velocities=include_player_velocities, annotate=annotate )
   mviz.plot_events( events.loc[frame:frame], figax = (fig,ax), indicators = ['Marker','Arrow'], annotate=False, color= 'k', alpha=1 )
   
   # plot pitch control surface
   if pass_team=='Home':
      cmap = 'bwr'
   else:
      cmap = 'bwr_r'
   ax.imshow(np.flipud(PPCF), extent=(-field_dimen[0]/2., field_dimen[0]/2., -field_dimen[1]/2., field_dimen[1]/2.),interpolation='spline36',vmin=0.0,vmax=1.0,cmap=cmap,alpha=0.5)

   return fig,ax

#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################

from scipy.optimize import fsolve


class player(object):

   # player object holds position, velocity
   def __init__(self,player_id,team,teamname,params={'amax':7,'vmax':5,'ttrl_sigma':0.54},xgrid=50,ygrid=32):
      self.id = player_id
      self.teamname = teamname
      self.playername = "%s_%s_" % (teamname,player_id)
      self.get_position(team)
      self.get_velocity(team)
      self.vmax = params['vmax']
      self.amax = params['amax']
      self.reaction_time = params['vmax']/params['amax']
      self.ttrl_sigma=params['ttrl_sigma'] # Standard deviation of sigmoid function in Spearman 2018 ('s') that determines uncertainty in player arrival time
      self.pitch_control = 0. # initialise this for later
      self.pitch_control_surface = np.zeros( shape = (ygrid, xgrid) )
      
   def get_position(self,team):
      self.position = np.array( [ team[self.playername+'x'], team[self.playername+'y'] ] )
      self.inframe = not np.any( np.isnan(self.position) )
      
   def get_velocity(self,team):
      self.velocity = np.array( [ team[self.playername+'vx'], team[self.playername+'vy'] ] )
      if np.any( np.isnan(self.velocity) ):
         self.velocity = np.array([0.,0.])

   def simple_time_to_reach_location(self,location):
    
      reaction_time=self.vmax/self.amax
      r_reaction = self.position + self.velocity*reaction_time
      arrival_time = reaction_time + np.linalg.norm(location-r_reaction)/self.vmax
      self.time_to_reach_location=arrival_time
      return(arrival_time)
   
   def improved_time_to_reach_location(self,location):
      Xf=location
      X0=self.position
      V0=self.velocity
      alpha = self.amax/self.vmax
      
      #equations of motion + equation 3 from assumption that the player accelerate 
      #with constant acceleration amax to vmax
      #we have to add abs(t) to make t be positive
      def equations(p):
         vxmax, vymax, t = p
         eq1 = Xf[0] - (X0[0] + vxmax*(abs(t) - (1 - np.exp(-alpha*abs(t)))/alpha)+((1 - np.exp(-alpha*abs(t)))/alpha)*V0[0])
         eq2 = Xf[1] - (X0[1] + vymax*(abs(t) - (1 - np.exp(-alpha*abs(t)))/alpha)+((1 - np.exp(-alpha*abs(t)))/alpha)*V0[1])
         eq3 = np.sqrt(vxmax**2+vymax**2) - self.vmax
         return (eq1,eq2,eq3)
      
      #prediction for three unknowns
      t_predict=np.linalg.norm(Xf-X0)/self.vmax+0.7
      v_predict=self.vmax*(Xf-X0)/np.linalg.norm(Xf-X0)
      vxmax, vymax, t =  fsolve(equations, (v_predict[0], v_predict[1], t_predict))

      self.time_to_reach_location=abs(t)
      
      return(abs(t))
   
   def probability_to_reach_location(self,T):
      f = 1/(1. + np.exp( -np.pi/np.sqrt(3.0)/self.ttrl_sigma * (T-self.time_to_reach_location ) ) )
      return f


def parameters():
   """
   default_model_params()
   
   Returns the default parameters that define and evaluate the model. See Spearman 2018 for more details.
   
   Parameters
   -----------
   time_to_control_veto: If the probability that another team or player can get to the ball and control it is less than 10^-time_to_control_veto, ignore that player.
   
   
   Returns
   -----------
   
   params: dictionary of parameters required to determine and calculate the model
   
   """
   # key parameters for the model, as described in Spearman 2018
   params = {}
   # model parameters
   params['amax'] = 7. # maximum player acceleration m/s/s
   params['vmax'] = 5. # maximum player speed m/s
   params['ttrl_sigma'] = 0.54 # Standard deviation of sigmoid function in Spearman 2018 ('s') that determines uncertainty in player arrival time
   params['kappa_def'] =  1. # kappa parameter in Spearman 2018 that gives the advantage defending players to control ball
   params['lambda_att'] = 3.99 # ball control parameter for attacking team
   params['lambda_def'] = 3.99 * params['kappa_def'] # ball control parameter for defending team
   params['average_ball_speed'] = 15. # average ball travel speed in m/s
   # numerical parameters for model evaluation
   params['int_dt'] = 0.04 # integration timestep (dt)
   params['max_int_time'] = 10 # upper limit on integral time
   params['model_converge_tol'] = 0.01 # assume convergence when PPCF>0.99 at a given location.
   # The following are 'short-cut' parameters. We do not need to calculated PPCF explicitly when a player has a sufficient head start. 
   # A sufficient head start is when the a player arrives at the target location at least 'time_to_control' seconds before the next player
   params['time_to_control_att'] = 3*np.log(10) * (np.sqrt(3)*params['ttrl_sigma']/np.pi + 1/params['lambda_att'])
   params['time_to_control_def'] = 3*np.log(10) * (np.sqrt(3)*params['ttrl_sigma']/np.pi + 1/params['lambda_def'])
   
   # sigma normal distribution for relevant pitch control
   params['sigma_normal'] = 23.9
   # alpha : dependence of the decision conditional probability by the PPCF
   params['alpha'] = 1.04
   
   return params


def initialise_players(team,teamname,params={'amax':7,'vmax':5},xgrid=50,ygrid=32):
   
   # get player  ids
   player_ids = np.unique( [ c.split('_')[1] for c in team.keys() if c[:4] == teamname ] )
   # create list
   team_players = []
   for p in player_ids:
      # create a player object for player_id 'p'
      team_player = player(p,team,teamname,params=params)
      if team_player.inframe:
         team_players.append(team_player)
         
   return team_players


def distance_to_location(location,player):
   origin=player.position
   return(np.linalg.norm(location-origin))


def calculate_basic_pitch_control_at_location(location, home_players, away_players, distance_function = distance_to_location):
    
   #initialise distance to location
   distance=np.inf
   closest_player=None
   closest_team=None
   
   for player in home_players:
      #calculate distance from player to location
      d=distance_function(location,player)
      if d<distance:
         closest_player=player.playername
         closest_team=player.teamname
         distance=d
         
   for player in away_players:
      #calculate distance from player to location
      d=distance_function(location,player)
      if d<distance:
         closest_player=player.playername
         closest_team=player.teamname
         distance=d
   
   if closest_team=='Home':
      #if Home team is closest --> probability to possess the ball=1 while for away time proba=0
      return(1,0)
   
   if closest_team=='Away':
      return(0,1)


def attacking_team_frame(events,frame):
    
   #the game doesn't start at frame 0, so we check that the frame asked is superior to the frame when game starts
   assert frame>=events['Start Frame'][0],'frame before game start'
   
   attacking_team=events[((events.Type=="RECOVERY") | (events.Type=="SET PIECE")) & (events['Start Frame']<=frame)]['Team'].values[-1] 

   return(attacking_team)


def where_home_team_attacks(home_team,events):
   '''
   Determines where teams attack on the first period using team x average position at game start
   
   Returns
   -------
      -1 if home team attacks on the left (x<0)
      1 if home team attacks on the right (x>0)
   
   '''
   game_start_frame=events.iloc[0]['Start Frame']
   home_team_x_cols=[c for c in home_team.columns if c.split('_')[-1]=='x' and c.split('_')[-2]!='ball']

   if home_team.loc[game_start_frame,home_team_x_cols].mean()>0:
      return(-1)
   else:
      return(1)


def find_offside_players(attacking_players, defending_players, where_attack, ball_pos):
   '''
   Determines which attacking players are in offside position. 
   A player is caught offside if he’s nearer to the opponent’s goal 
   than both the ball and the second-last opponent (including the goalkeeper).
   
   Returns
   -------
      offside_players : the list of offside players names
   '''
   
   offside_players=[]
   
   # if attacking team attacks on the right
   if where_attack==1:
      
      #find the second-last defender
      x_defending_players=[]
      for player in defending_players:
         x_defending_players.append(player.position[0])
      x_defending_players=np.sort(x_defending_players)
      second_last_defender_x=x_defending_players[-2]
      
      for player in attacking_players:
         position=player.position
         #if player is nearer to the opponent's goal than the ball
         if position[0]>ball_pos[0] and position[0]>second_last_defender_x:
               offside_players.append(player)
               
   # if attacking team attacks on the right
   if where_attack==-1:
      
      #find the second-last defender
      x_defending_players=[]
      for player in defending_players:
         x_defending_players.append(player.position[0])
      x_defending_players=np.sort(x_defending_players)
      second_last_defender_x=x_defending_players[1]
      
      for player in attacking_players:
         position=player.position
         #if player is nearer to the opponent's goal than the ball
         if position[0]<ball_pos[0] and position[0]<second_last_defender_x:
               offside_players.append(player.playername)
               
   return(offside_players)


def calculate_pitch_control_at_target(target_position, attacking_players, defending_players, ball_start_pos, where_attack, params):
   """ calculate_pitch_control_at_target
   
   Calculates the pitch control probability for the attacking and defending teams at a specified target position on the ball.
   
   Parameters
   -----------
      target_position: size 2 numpy array containing the (x,y) position of the position on the field to evaluate pitch control
      attacking_players: list of 'player' objects (see player class above) for the players on the attacking team (team in possession)
      defending_players: list of 'player' objects (see player class above) for the players on the defending team
      ball_start_pos: Current position of the ball (start position for a pass). If set to NaN, function will assume that the ball is already at the target position.
      where_attack: where attacking team attacks (1 on the right, -1 on the left)
      params: Dictionary of model parameters
      
   Returns
   -----------
      PPCFatt: Pitch control probability for the attacking team
      PPCFdef: Pitch control probability for the defending team ( 1-PPCFatt-PPCFdef <  params['model_converge_tol'] )
   """
   
   # calculate ball travel time from start position to end position.
   if ball_start_pos is None or any(np.isnan(ball_start_pos)): # assume that ball is already at location
      ball_travel_time = 0.0 
   else:
      # ball travel time is distance to target position from current ball position divided assumed average ball speed
      ball_travel_time = np.linalg.norm( target_position - ball_start_pos )/params['average_ball_speed']
      
   # find offside attacking players
   offside_players=find_offside_players(attacking_players, defending_players, where_attack, ball_start_pos)
   
   # first get arrival time of 'nearest' attacking player (nearest also dependent on current velocity) (if player isn't offside)
   tau_min_att = np.nanmin( [p.improved_time_to_reach_location(target_position) for p in attacking_players if p.playername not in offside_players] )
   tau_min_def = np.nanmin( [p.improved_time_to_reach_location(target_position) for p in defending_players] )
   # check whether we actually need to solve equation 
   if tau_min_att-max(ball_travel_time,tau_min_def) >= params['time_to_control_def']:
      # if defending team can arrive significantly before attacking team, no need to solve pitch control model
      return 0., 1.
   elif tau_min_def-max(ball_travel_time,tau_min_att) >= params['time_to_control_att']:
      # if attacking team can arrive significantly before defending team, no need to solve pitch control model
      return 1., 0.
   else: 
      # solve pitch control model by integrating equation 3 in Spearman et al.
      # first remove any player that is far (in time) from the target location
      # remove offside players
      attacking_players = [p for p in attacking_players if (p.playername not in offside_players) and (p.time_to_reach_location-tau_min_att < params['time_to_control_att']) ]
      defending_players = [p for p in defending_players if p.time_to_reach_location-tau_min_def < params['time_to_control_def'] ]
      
      # set up integration arrays
      dT_array = np.arange(ball_travel_time-params['int_dt'],ball_travel_time+params['max_int_time'],params['int_dt']) 
      PPCFatt = np.zeros_like( dT_array )
      PPCFdef = np.zeros_like( dT_array )
      
      # set PPCF to 0. for each player
      for player in attacking_players:
         player.pitch_control=0.
      for player in defending_players:
         player.pitch_control=0.
      
      # integration equation 3 of Spearman 2018 until convergence or tolerance limit hit (see 'params')
      ptot = 0.0
      i = 1
      
      while 1-ptot>params['model_converge_tol'] and i<dT_array.size: 
         T = dT_array[i]
         for player in attacking_players:
               
               # calculate lambda for 'player' (0 if offside)
               if player.playername in offside_players:
                  lambda_att=0
               else:
                  lambda_att=params['lambda_att']
                  
               # calculate ball control probablity for 'player' in time interval T+dt
               dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_to_reach_location( T ) * lambda_att
               
               # make sure it's greater than zero
               assert dPPCFdT>=0, 'Invalid attacking player probability (calculate_pitch_control_at_target)'

               player.pitch_control += dPPCFdT*params['int_dt'] # total contribution from individual player
               PPCFatt[i] += player.pitch_control # add to sum over players in the attacking team (remembering array element is zero at the start of each integration iteration)
               
         for player in defending_players:
               # calculate ball control probablity for 'player' in time interval T+dt
               dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_to_reach_location( T ) * params['lambda_def']
               
               # make sure it's greater than zero
               assert dPPCFdT>=0, 'Invalid defending player probability (calculate_pitch_control_at_target)'

               player.pitch_control += dPPCFdT*params['int_dt'] # total contribution from individual player
               PPCFdef[i] += player.pitch_control # add to sum over players in the defending team
               
         ptot = PPCFdef[i]+PPCFatt[i] # total pitch control probability 
         i += 1
      if i>=dT_array.size:
         print("Integration failed to converge: %1.3f" % (ptot) )
         
      return PPCFatt[i-1], PPCFdef[i-1]


def generate_pitch_control_for_frame(frame, tracking_home, tracking_away, events, params, field_dimen = (106.,68.,), n_grid_cells_x = 50, return_players=False):
   """ generate_pitch_control_for_frame
   
   Evaluates pitch control surface over the entire field at the moment of the given frame
   
   Parameters
   -----------
      frame: instant at which the pitch control surface should be calculated
      tracking_home: tracking DataFrame for the Home team
      tracking_away: tracking DataFrame for the Away team
      events: Dataframe containing the event data
      params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
      field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
      n_grid_cells_x: Number of pixels in the grid (in the x-direction) that covers the surface. Default is 50.
                     n_grid_cells_y will be calculated based on n_grid_cells_x and the field dimensions
      
   Returrns
   -----------
      PPCFa: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team.
            Surface for the defending team is just 1-PPCFa.
      xgrid: Positions of the pixels in the x-direction (field length)
      ygrid: Positions of the pixels in the y-direction (field width)
   """
   
   # get the details of the event (team in possession, ball_start_position, where team in possession attacks)
   attacking_team = attacking_team_frame(events,frame)
   assert attacking_team=='Home' or attacking_team=='Away', 'attacking team should be Away or Home'
   
   ball_start_pos = np.array( [ tracking_home.loc[frame]['ball_x'], tracking_home.loc[frame]['ball_y'] ] )
   
   where_home_attacks = where_home_team_attacks(tracking_home,events)
   period=tracking_home.loc[frame]['Period']
   if attacking_team=='Home':
      if period==1:
         where_attack=where_home_attacks
      else:
         where_attack=-where_home_attacks
   else:
      if period==1:
         where_attack=-where_home_attacks
      else:
         where_attack=where_home_attacks
      
   # break the pitch down into a grid
   n_grid_cells_y = int(n_grid_cells_x*field_dimen[1]/field_dimen[0])
   xgrid = np.linspace( -field_dimen[0]/2., field_dimen[0]/2., n_grid_cells_x)
   ygrid = np.linspace( -field_dimen[1]/2., field_dimen[1]/2., n_grid_cells_y )
   
   # initialise pitch control grids for attacking and defending teams 
   PPCFa = np.zeros( shape = (len(ygrid), len(xgrid)) )
   PPCFd = np.zeros( shape = (len(ygrid), len(xgrid)) )
   
   # initialise player positions and velocities for pitch control calc (so that we're not repeating this at each grid cell position)
   if attacking_team=='Home':
      attacking_players = initialise_players(tracking_home.loc[frame],'Home',params, xgrid=len(xgrid), ygrid=len(ygrid))
      defending_players = initialise_players(tracking_away.loc[frame],'Away',params, xgrid=len(xgrid), ygrid=len(ygrid))
   else:
      defending_players = initialise_players(tracking_home.loc[frame],'Home',params, xgrid=len(xgrid), ygrid=len(ygrid))
      attacking_players = initialise_players(tracking_away.loc[frame],'Away',params, xgrid=len(xgrid), ygrid=len(ygrid))

   # calculate pitch control model at each location on the pitch
   # if we want to save individual pitch control
   if return_players:
      for i in range( len(ygrid) ):
         for j in range( len(xgrid) ):
               target_position = np.array( [xgrid[j], ygrid[i]] )
               PPCFa[i,j],PPCFd[i,j] = calculate_pitch_control_at_target(target_position, attacking_players, defending_players, ball_start_pos, where_attack, params)
               for player in attacking_players:
                  player.pitch_control_surface[i,j] = player.pitch_control
               for player in defending_players:
                  player.pitch_control_surface[i,j] = player.pitch_control
   else:
      for i in range( len(ygrid) ):
         for j in range( len(xgrid) ):
               target_position = np.array( [xgrid[j], ygrid[i]] )
               PPCFa[i,j],PPCFd[i,j] = calculate_pitch_control_at_target(target_position, attacking_players, defending_players, ball_start_pos, where_attack, params)

   # check probabilitiy sums within convergence
   checksum = np.sum( PPCFa + PPCFd ) / float(n_grid_cells_y*n_grid_cells_x ) 
   assert abs(1-checksum) < params['model_converge_tol'], "Checksum failed: %1.3f" % (1-checksum)
   
   if return_players:
      players_pitch_control = {}
      for player in attacking_players:
         players_pitch_control[player.playername] = player.pitch_control_surface
      for player in defending_players:
         players_pitch_control[player.playername] = player.pitch_control_surface
         
      return(PPCFa,xgrid,ygrid,players_pitch_control)
   
   return PPCFa,xgrid,ygrid

def plot_pitch_control_for_frame(frame, tracking_home, tracking_away, events, params, alpha = 0.7, include_player_velocities=True, annotate=True, field_dimen = (106.,68.,), n_grid_cells_x = 50):
   """ plot_pitch_control_for_frame(frame, tracking_home, tracking_away, events, params ,PPCF, xgrid, ygrid )
   
   Plots the pitch control surface at the instant of the frame. Player and ball positions are overlaid.
   
   Parameters
   -----------
      frame: the instant at which the pitch control surface should be calculated
      events: Dataframe containing the event data
      tracking_home: (entire) tracking DataFrame for the Home team
      tracking_away: (entire) tracking DataFrame for the Away team
      PPCF: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team (as returned by the generate_pitch_control_for_event in Metrica_PitchControl)
      xgrid: Positions of the pixels in the x-direction (field length) as returned by the generate_pitch_control_for_event in Metrica_PitchControl
      ygrid: Positions of the pixels in the y-direction (field width) as returned by the generate_pitch_control_for_event in Metrica_PitchControl
      alpha: alpha (transparency) of player markers. Default is 0.7
      include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
      annotate: Boolean variable that determines with player jersey numbers are added to the plot (default is False)
      field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
      
   Returns
   -----------
      fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
   """    
   # plot frame and event
   fig,ax = mviz.plot_pitch(field_color='white', field_dimen = field_dimen)
   mviz.plot_frame( tracking_home.loc[frame], tracking_away.loc[frame], figax=(fig,ax), PlayerAlpha=alpha, include_player_velocities=include_player_velocities, annotate=annotate )
   
   #generate pitch control
   PPCF, xgrid, ygrid=generate_pitch_control_for_frame(frame, tracking_home, tracking_away, events, params, field_dimen = field_dimen, n_grid_cells_x = n_grid_cells_x)
   
   #find attacking team
   attacking_team = attacking_team_frame(events,frame)
   
   # plot pitch control surface
   if attacking_team=='Home':
      cmap = 'bwr'
   else:
      cmap = 'bwr_r'
   ax.imshow(np.flipud(PPCF), extent=(np.amin(xgrid), np.amax(xgrid), np.amin(ygrid), np.amax(ygrid)),interpolation='hanning',vmin=0.0,vmax=1.0,cmap=cmap,alpha=0.5)
   
   return fig,ax


# def basic_pitch_control_for_frame(tracking_home,tracking_away,frame, field_dimen = (106.,68.,), n_grid_cells_x = 50, distance_function = distance_to_location):
    
#    # break the pitch down into a grid
#    n_grid_cells_y = int(n_grid_cells_x*field_dimen[1]/field_dimen[0])
#    xgrid = np.linspace( -field_dimen[0]/2., field_dimen[0]/2., n_grid_cells_x)
#    ygrid = np.linspace( -field_dimen[1]/2., field_dimen[1]/2., n_grid_cells_y )
   
#    #intialise pitch control
#    pitch_control_home = np.zeros( shape = (len(ygrid), len(xgrid)) )
#    pitch_control_away = np.zeros( shape = (len(ygrid), len(xgrid)) )
   
#    # initialise player positions and velocities for pitch control calc (so that we're not repeating this at each grid cell position)
#    home_players = initialise_players(tracking_home.loc[frame],'Home')
#    away_players = initialise_players(tracking_away.loc[frame],'Away')
   
#    # calculate pitch control at each location on the pitch
#    for i in range( len(ygrid) ):
#       for j in range( len(xgrid) ):
#          location = np.array( [xgrid[j], ygrid[i]] )
         
#          # we calculate pitch control for home and away team because we will need it for more advance 
#          # pitch control model, but it useless with basic pitch control
#          pitch_control_home[i,j],pitch_control_away[i,j] = calculate_basic_pitch_control_at_location(location, home_players, away_players, distance_function = distance_function)
         
#    return pitch_control_home,xgrid,ygrid

# def plot_basic_pitch_control_for_frame(tracking_home,tracking_away,frame, figax=None, field_dimen = (106.,68.,), field_color='white', n_grid_cells_x = 50, alpha = 0.7, include_player_velocities=True, annotate=True, distance_function = distance_to_location):
    
#    if figax is None: # create new pitch 
#       fig,ax = mviz.plot_pitch(field_dimen=field_dimen, field_color= field_color)
#    else: # overlay on a previously generated pitch
#       fig,ax = figax # unpack tuple
   
#    pitch_control,xgrid,ygrid=basic_pitch_control_for_frame(tracking_home,tracking_away, frame, field_dimen = field_dimen, n_grid_cells_x = n_grid_cells_x, distance_function = distance_function)
   
#    #plot frame
#    mviz.plot_frame( tracking_home.loc[frame], tracking_away.loc[frame] , figax=(fig,ax), PlayerAlpha=alpha, include_player_velocities=include_player_velocities, annotate=annotate)
   
#    # plot pitch control surface
#    cmap = 'bwr'
#    im=ax.imshow(np.flipud(pitch_control), extent=(np.amin(xgrid), np.amax(xgrid), np.amin(ygrid), np.amax(ygrid)),vmin=0.0,vmax=1.0,cmap=cmap,alpha=0.5)
#    fig.colorbar(im)
   
#    return fig,ax