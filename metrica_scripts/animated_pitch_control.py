import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from metrica_scripts import Metrica_IO as mio
from metrica_scripts import Metrica_Viz as mviz
from metrica_scripts import Metrica_Velocities as mvel

from moviepy import editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage

class player(object):
   # player object holds position, velocity
   def __init__(self,player_id,team,teamname,params={'amax':7,'vmax':5}):
      self.id = player_id
      self.teamname = teamname
      self.playername = "%s_%s_" % (teamname,player_id)
      self.get_position(team)
      self.get_velocity(team)
      self.vmax = params['vmax']
      self.amax = params['amax']
      self.reaction_time = params['vmax']/params['amax']
      self.pitch_control = 0. # initialise this for later
      
   def get_position(self,team):
      self.position = np.array( [ team[self.playername+'x'], team[self.playername+'y'] ] )
      self.inframe = not np.any( np.isnan(self.position) )
      
   def get_velocity(self,team):
      self.velocity = np.array( [ team[self.playername+'vx'], team[self.playername+'vy'] ] )
      if np.any( np.isnan(self.velocity) ):
         self.velocity = np.array([0.,0.])


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


def basic_pitch_control_for_frame(tracking_home,tracking_away,frame, field_dimen = (106.,68.,), n_grid_cells_x = 50, distance_function = distance_to_location):
    
   # break the pitch down into a grid
   n_grid_cells_y = int(n_grid_cells_x*field_dimen[1]/field_dimen[0])
   xgrid = np.linspace( -field_dimen[0]/2., field_dimen[0]/2., n_grid_cells_x)
   ygrid = np.linspace( -field_dimen[1]/2., field_dimen[1]/2., n_grid_cells_y )
   
   #intialise pitch control
   pitch_control_home = np.zeros( shape = (len(ygrid), len(xgrid)) )
   pitch_control_away = np.zeros( shape = (len(ygrid), len(xgrid)) )
   
   # initialise player positions and velocities for pitch control calc (so that we're not repeating this at each grid cell position)
   home_players = initialise_players(tracking_home.loc[frame],'Home')
   away_players = initialise_players(tracking_away.loc[frame],'Away')
   
   # calculate pitch control at each location on the pitch
   for i in range( len(ygrid) ):
      for j in range( len(xgrid) ):
         location = np.array( [xgrid[j], ygrid[i]] )
         
         # we calculate pitch control for home and away team because we will need it for more advance 
         # pitch control model, but it useless with basic pitch control
         pitch_control_home[i,j],pitch_control_away[i,j] = calculate_basic_pitch_control_at_location(location, home_players, away_players, distance_function = distance_function)
         
   return pitch_control_home,xgrid,ygrid


def plot_basic_pitch_control_for_frame(tracking_home,tracking_away,frame, figax=None, field_dimen = (106.,68.,), field_color='white', n_grid_cells_x = 50, alpha = 0.7, include_player_velocities=True, annotate=True, distance_function = distance_to_location):
    
   if figax is None: # create new pitch 
      fig,ax = mviz.plot_pitch(field_dimen=field_dimen, field_color= field_color)
   else: # overlay on a previously generated pitch
      fig,ax = figax # unpack tuple
   
   pitch_control,xgrid,ygrid=basic_pitch_control_for_frame(tracking_home,tracking_away, frame, field_dimen = field_dimen, n_grid_cells_x = n_grid_cells_x, distance_function = distance_function)
   
   #plot frame
   mviz.plot_frame( tracking_home.loc[frame], tracking_away.loc[frame] , figax=(fig,ax), PlayerAlpha=alpha, include_player_velocities=include_player_velocities, annotate=annotate)
   
   # plot pitch control surface
   cmap = 'bwr'
   im=ax.imshow(np.flipud(pitch_control), extent=(np.amin(xgrid), np.amax(xgrid), np.amin(ygrid), np.amax(ygrid)),vmin=0.0,vmax=1.0,cmap=cmap,alpha=0.5)
   fig.colorbar(im)
   
   return fig,ax


def draw_frame_x(hometeam,awayteam, t, players_to_annotate, figax=None, fps=25, 
                 team_colors=('r','b'), field_dimen = (106.0,68.0), 
                 include_player_velocities=True, PlayerMarkerSize=10, PlayerAlpha=0.7,dpi=100):
    
   #we have to convert t*fps to int because t is a float
   fig,ax=mviz.plot_pitch(field_color='white')
   fig,ax=mviz.plot_frame(hometeam.iloc[int(t*fps)], awayteam.iloc[int(t*fps)], figax=(fig,ax), 
                        team_colors=team_colors, field_dimen = field_dimen, 
                        include_player_velocities=include_player_velocities, PlayerMarkerSize=PlayerMarkerSize, PlayerAlpha=PlayerAlpha)
   
   fig,ax=plot_basic_pitch_control_for_frame(hometeam,awayteam,int(t*fps),figax=(fig,ax))
   
   players_id=players_to_annotate.iloc[:,0].values
   nums=players_to_annotate.iloc[:,1].values
   for k in range(len(nums)):
      ax.text(hometeam.iloc[int(t*fps)]['Home_'+str(players_id[k])+'_x'],hometeam.iloc[int(t*fps)]['Home_'+str(players_id[k])+'_y'],str(int(nums[k])),fontsize=10,color='k',horizontalalignment='center', verticalalignment='center')
   
   image = mplfig_to_npimage(fig)
   plt.close()
   return image

def make_animation(hometeam,awayteam, fpath, players_to_annotate, fname='clip_test', figax=None, fps=25, 
                   team_colors=('r','b'), field_dimen = (106.0,68.0), 
                   include_player_velocities=True, PlayerMarkerSize=15, PlayerAlpha=0.7,dpi=100):
    
   length=len(hometeam)/fps

   clip = mpy.VideoClip(
      lambda x: draw_frame_x(hometeam, awayteam, x, players_to_annotate, figax=figax, team_colors=team_colors, field_dimen = field_dimen, include_player_velocities=include_player_velocities, PlayerMarkerSize=PlayerMarkerSize, PlayerAlpha=PlayerAlpha),
      duration=length).set_fps(fps)

   clip.write_videofile(fpath+fname+'.mp4')