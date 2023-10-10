from pathlib import Path

class Config:

    # Construction des dossiers de travail
    ASSETS_PATH = Path('../ASSETS')
    DATASET_PATH = ASSETS_PATH / 'DATA'
    PLUIE_ETP_PATH = DATASET_PATH / 'RAW' / 'ETP_PLUIE'
    SPEI_PATH = DATASET_PATH / 'RAW' / 'SPEI'
    ERA5_PATH = DATASET_PATH / 'RAW' / 'ERA5'
    STATION_PATH = DATASET_PATH / 'RAW' / 'INFO_STATIONS'
    MODELS_PATH = ASSETS_PATH / 'models'
    FEATURES_PATH = ASSETS_PATH / 'features'
    METRICS_FILE_PATH = ASSETS_PATH / 'metrics'
    FILES_TRAITED_PATH=DATASET_PATH / 'PREPARED'
    ERA5_DATA_PATH = DATASET_PATH / 'ERA5'
    DATASET_DIR=DATASET_PATH / 'DATASET'

    #Partie liée au processing des données de era5
    #Fichier de coordonnées des stations
    
    #Fichiers
    PLUIE_FILE_NAME='pluie_jour.xls'           #Données journalières de pluies
    ETP_FILE_NAME='etp_jour.xls'
    STAT_COORD_FILE_NAME='station_coord.xlsx'  #Coordonnées des stations
    SPEI_1DEK_FILE_NAME='SPEI_1dek.csv'
    SPEI_1MON_FILE_NAME='SPEI_1mon.csv'
    SPEI_3MON_FILE_NAME='SPEI_3mon.csv'
    SPEI_6MON_FILE_NAME='SPEI_6mon.csv'              #Données journalières de l'evapotranspiration
    PLUIE_ETP_FILE_NAME='merged_data.csv'      #Fusionner les deux fichiers en un après traitement
    STATION_TEST='boromo.csv'                  #Permettre de sauvegarder une station test pour controle
    STATION_ALL='all_station.csv'
    DATA_1DEK_EXTR='data_ext_1dek.csv'
    DATA_1DEK_SEV='data_sev_1dek.csv'
    DATA_1MON_EXTR='data_ext_1mon.csv'
    DATA_1MON_SEV='data_sev_1mon.csv'
    DATA_3MON_EXTR='data_ext_3mon.csv'
    DATA_3MON_SEV='data_sev_3mon.csv'
    DATA_6MON_EXTR='data_ext_6mon.csv'
    DATA_6MON_SEV='data_sev_6mon.csv'
    #Les fichiers dans era5
    WIND_V_975='V_wind_975.nc' #'v_wind_1.nc'
    WIND_V_975_CSV='v_wind_975.csv' # 'v_wind_1.csv'
    WIND_V_975_DEK='v_wind_975_dek.csv'
    WIND_V_975_MON='v_wind_975_mon.csv'
    
    WIND_U_700='U_wind_700.nc' #'v_wind_1.nc'
    WIND_U_700_CSV='u_wind_700.csv' # 'v_wind_1.csv'
    WIND_U_700_DEK='u_wind_700_dek.csv'
    WIND_U_700_MON='u_wind_700_mon.csv'  
    
    WIND_U_100='U_wind_100.nc' #'v_wind_1.nc'
    WIND_U_100_CSV='u_wind_100.csv' # 'v_wind_1.csv'
    WIND_U_100_DEK='u_wind_100_dek.csv'
    WIND_U_100_MON='u_wind_100_mon.csv' 
    
    #EAU_PREC='cloud_liquid_water_flux.nc' #'v_wind_1.nc'
    EAU_PREC='eau_precipitable.nc'
    EAU_PREC_CSV='eau_prec.csv' # 'v_wind_1.csv'
    EAU_PREC_CSV_DEK='eau_prec_dek.csv'
    EAU_PREC_CSV_MON='eau_prec_mon.csv'    
    
    #POINT_ROSEE='drew_point.nc' #'v_wind_1.nc'
    POINT_ROSEE='point_rosee.nc'
    POINT_ROSEE_CSV='point_rosee.csv' # 'v_wind_1.csv'
    POINT_ROSEE_CSV_DEK='point_rosee_dek.csv'
    POINT_ROSEE_CSV_MON='point_rosee_mon.csv' 
    
    SOL_WATERPOINT='vol_soil_water.nc'
    SOL_WATERPOINT_CSV='vol_soil_water.csv' # 'v_wind_1.csv'
    SOL_WATERPOINT_CSV_DEK='vol_soil_water_dek.csv'
    SOL_WATERPOINT_CSV_MON='vol_soil_water_mon.csv'      
    
    #Dossiers de données de temperature de mer
    SST_LEF_PATH= ERA5_PATH / 'SST_LEF_DEK'
    SST_NINO_PATH= ERA5_PATH / 'SST_NINO_DEK'
    SST_LEF_SP_MN_PATH= ERA5_PATH / 'SST_LEF_FLDMEAN'   # Recupère les fichiers temp langue d'eau froide avec les moyennes spatiales
    SST_NINO_SP_MN_PATH=ERA5_PATH / 'SST_NINO_FLDMEAN'
    SST_LEF_MERGE='sst_leg_merged.nc'
    SST_NINO_MERGE='sst_nino_merged.nc'
    
    #Fichier pour sauvegarder les parametres issus de ERA5
    ERA_FUSION_DEK='era_decade.csv'
    ERA_FUSION_MON='era_mensuel.csv'
    
    
class Constant:
    #Labels secheresse
    LABEL_1='Humidité extrême'
    LABEL_2='Humidité sévère'
    LABEL_3='Humidité modérée'
    LABEL_4='Normale'
    LABEL_5='Sécheresse modérée'
    LABEL_6='Sécheresse sévère'
    LABEL_7='Sécheresse extrême'
    #Seuils secheresse 
    SEUIL_1=-1.0
    SEUIL_2=-1.5
    SEUIL_3=-1.99
    SEUIL_4=-2.0
    # Partage des datasets avant l'apprentissage 
    TEST_SIZE=0.2
    RANDOM_SEED=42
    
    # Mois d'avance pour la prediction
    MONTH_AVANT=2


