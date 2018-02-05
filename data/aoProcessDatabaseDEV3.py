#!/usr/bin/python -u
# encoding: utf-8
########################################################################################################################
from __future__ import print_function
import re
import os
import argparse
import platform
import zipfile
import numpy as np
import contextlib


########################################################################################################################
def aoDatabasePath2(aodir):
    if os.path.isdir(aodir + '/AutoOptiProcessChain'):
        aodir = os.path.abspath(aodir + '/AutoOptiProcessChain')
    else:
        aodir = os.path.abspath(aodir)

    DatabaseCSV = aodir + '/master/Output/database.csv'
    ResultNames = aodir + '/master/Input/ResultNames'
    GenDataInput = aodir + '/master/Input/genData.input'

    if os.path.isfile(DatabaseCSV) and os.path.isfile(ResultNames) and os.path.isfile(GenDataInput):
        return DatabaseCSV, ResultNames, GenDataInput
    else:
        return None


def aoDatabaseCheck(aodir):
    if aoDatabasePath2(aodir) is None:
        return False
    else:
        return True


def aoDatabasePath(aodir):
    aodir = os.path.abspath(aodir)
    if os.path.isfile('%s/master/Output/database.csv' % aodir):
        return '%s/master/Output/database.csv' % aodir
    elif os.path.isfile('%s/AutoOptiProcessChain/master/Output/database.csv' % aodir):
        return '%s/AutoOptiProcessChain/master/Output/database.csv' % aodir
    else:
        return None


def aoImportInput(aodir):
    flowparam_names = []
    flowparam_roi = []
    fitness_roi = []

    if os.path.isdir(aodir + '/AutoOptiProcessChain'):
        aodir = os.path.abspath(aodir + '/AutoOptiProcessChain')
    else:
        aodir = os.path.abspath(aodir)

    with open(aodir + '/master/Input/ResultNames', 'r') as fi:
        for i, line in enumerate(fi.readlines()):
            line = re.sub('^\s*\d+\s+(.+)\n', r'\1', line)
            flowparam_names.append(line)
        flowparam_names.append('Placeholder')

    with open(aodir + '/master/Input/genData.input', 'r') as fi:
        lines = fi.readlines()
        for i, line in enumerate(lines):
            if re.match('fitness', line):
                ind = int(re.sub('.*flow_(\d+).*', r'\1', line))
                if re.match('.*-flow_\d+.*', line):
                    fitness_roi.append(['mfitness', ind])
                else:
                    fitness_roi.append(['pfitness', ind])
            elif re.match('REGIONOFINTEREST\s*\t*1', line):
                k = 0
                for j in range(i, len(lines)):
                    if (re.match('-*\d*\.*\d+\s*\t*-*\d*\.*\d+', lines[j]) or 
                    re.match('-inf\s*\t*-*\d*\.*\d+', lines[j]) or
                    re.match('-*\d*\.*\d+\s*\t*inf', lines[j]) or
                    re.match('-inf\s*\t*inf', lines[j])):
                        minmax = re.split('\s+', lines[j])
                        if k < len(fitness_roi):
                            fitness_roi[k].append(float(minmax[0]))
                            fitness_roi[k].append(float(minmax[1]))
                            k += 1
                        else:
                            break
            elif re.match('FLOWPARAM_ROI\s+\d+', line):
                k = 0
                n_flowparam = int(re.sub('FLOWPARAM_ROI\s+(\d+).+', r'\1', line))
                for j in range(i, len(lines)):
                    if re.match('\d+\s+-*\d*\.*\d+\s+-*\d*\.*\d+', lines[j]):
                        idminmax = re.split('\s+', lines[j])
                        if k < n_flowparam:
                            flowparam_roi.append(
                                ['flowparam', int(idminmax[0]), float(idminmax[1]), float(idminmax[2])])
                            k += 1
                        else:
                            break

    return flowparam_names, flowparam_roi, fitness_roi


def aoImportDatabase(aodir):
    DB = {
        '$header': ['ID', 'TIME', 'TIME_DB', 'memberTime'],
        '$header_flowparam': [],
        '$order': [],
        '$fitness': [],
        '$marker_database': [],
        '$marker_converged': [],
        '$marker_valid': [],
        '$marker_valid_fitness': [],
        '$marker_valid_flowparam': [],
        '$marker_pareto': [],
        '$marker_hifi': [],
        '$marker_lofi': [],
    }
    data = []
    header = []
    flowparam_names, flowparam_roi, fitness_roi = aoImportInput(aodir)
    
    DB['$header_flowparam'] = flowparam_names
   
    for item in fitness_roi:
        DB['$fitness'].append(item[1])
    
    with open(aoDatabasePath(aodir), 'r') as fi:
        for i, line in enumerate(fi.readlines()):
            line = re.sub(',\t\n', '', line)
            param = re.split(',\t', line)
            if param[0] == 'postionInDataBase':
                header = list(param)
            else:
                data.append([1234567890 if x == 'nan' else float(x) for x in param])
    if len(data)==0:
    	return DB, header
    data = zip(*data)

    for i, _header in enumerate(header):
        if re.match('flowparam\d+', _header):
            ind = int(re.sub('flowparam', '', _header))
            if not re.match('(unknown|Placeholder)', flowparam_names[ind]):
                name = flowparam_names[ind] + (str(i) if flowparam_names[ind] in DB['$header'] else '')
                DB[name] = list(data[i])
                DB['$header'].append(name)
            for roi in fitness_roi + flowparam_roi:
                if roi[1] == ind:
                    DB['%s%d' % (roi[0], roi[1])] = list(data[i])
                    DB['%s%d_min' % (roi[0], roi[1])] = [roi[2]] * len(data[i])
                    DB['%s%d_max' % (roi[0], roi[1])] = [roi[3]] * len(data[i])
        else:
            DB[_header] = list(data[i])
            DB['$header'].append(_header)

    for roi in fitness_roi + flowparam_roi:
        DB['$header'].append('%s%d' % (roi[0], roi[1]))
        DB['$header'].append('%s%d_min' % (roi[0], roi[1]))
        DB['$header'].append('%s%d_max' % (roi[0], roi[1]))

    for roi, fidelity, pareto in zip(DB['distToRegOfInt'], DB['fidelity'], DB['ParetoRank']):
        DB['$marker_database'].append(1)
        DB['$marker_converged'].append(1 if roi < 7000 else 0)
        DB['$marker_valid'].append(1 if roi == 0 else 0)
        DB['$marker_valid_fitness'].append(1)
        DB['$marker_valid_flowparam'].append(1)
        DB['$marker_pareto'].append(1 if roi < 7000 and pareto == 1 else 0)
        DB['$marker_hifi'].append(1 if roi < 7000 and fidelity == 0 else 0)
        DB['$marker_lofi'].append(1 if roi < 7000 and fidelity == 1 else 0)

    for roi in fitness_roi + flowparam_roi:
        for ind, _ in enumerate(DB['$marker_valid_flowparam']):
            roi_name = roi[0]
            roi_val = DB['%s%d' % (roi[0], roi[1])][ind]
            roi_min = DB['%s%d_min' % (roi[0], roi[1])][ind]
            roi_max = DB['%s%d_max' % (roi[0], roi[1])][ind]
            if roi_name == 'flowparam' and (roi_val < roi_min or roi_val > roi_max):
                DB['$marker_valid_flowparam'][ind] *= 0
            elif roi_name == 'pfitness' and (roi_val < roi_min or roi_val > roi_max):
                DB['$marker_valid_fitness'][ind] *= 0
            elif roi_name == 'mfitness' and (-roi_val < roi_min or -roi_val > roi_max):
                DB['$marker_valid_fitness'][ind] *= 0

    DB['$order'] = [-1] * (int(max(DB['memberNumber']))+1)
    for ind, member in enumerate(DB['memberNumber']):    	
        DB['$order'][int(member)] = ind
        
    #missing items from the database are not treated properly
    
    
    DB['ID'] = [0] * sum(DB['$marker_database'])
    member = 0
    for ind, ind_order in enumerate(DB['$order']):
        converged = DB['$marker_converged'][ind_order]
        DB['ID'][ind_order] = member if converged else -ind
        member += 1 if converged else 0

    DB['memberTime'] = [0] * sum(DB['$marker_database'])
    for ind in DB['$order']:
        if ind<0:   continue
        for key in DB['$header']:
            if re.match('timeOfProcess.+', key):
                DB['memberTime'][ind] += float(DB[key][ind])

    DB['TIME'] = [0] * sum(DB['$marker_database'])
    DB['TIME_DB'] = [0] * sum(DB['$marker_database'])
    time = 0
    time_db = 0
    for ind in DB['$order']:
        if ind<0:   continue
        DB['TIME_DB'][ind] = time_db
        time_db += float(DB['memberTime'][ind]) / 3600
        if DB['$marker_converged'][ind]:
            DB['TIME'][ind] = time
            time += float(DB['memberTime'][ind]) / 3600
        else:
            DB['TIME'][ind] = 0

    return DB, header


def aoDatabaseConvergence(DB, sample=150):
    DB['$header'] += ['paretoConverged', 'paretoValid', 'paretoValidFitness', 'paretoValidFlowparam',
                      'residualConverged', 'residualValid', 'residualValidFitness', 'residualValidFlowparam']

    DB['paretoConverged'] = [0] * sum(DB['$marker_database'])
    DB['paretoValid'] = [0] * sum(DB['$marker_database'])
    DB['paretoValidFitness'] = [0] * sum(DB['$marker_database'])
    DB['paretoValidFlowparam'] = [0] * sum(DB['$marker_database'])

    DB['residualConverged'] = [0] * sum(DB['$marker_database'])
    DB['residualValid'] = [0] * sum(DB['$marker_database'])
    DB['residualValidFitness'] = [0] * sum(DB['$marker_database'])
    DB['residualValidFlowparam'] = [0] * sum(DB['$marker_database'])

    pareto_converged = 0
    pareto_valid = 0
    pareto_valid_fitness = 0
    pareto_valid_flowparam = 0
    
    for ind in DB['$order']:
        if ind<0:   continue
        if DB['$marker_hifi'][ind]:
            pareto_converged = DB['fitness0'][ind]
            pareto_valid = DB['fitness0'][ind]
            pareto_valid_fitness = DB['fitness0'][ind]
            pareto_valid_flowparam = DB['fitness0'][ind]
            break

    for ind in DB['$order']:
        if ind<0:   continue
        if DB['$marker_hifi'][ind]:
            pareto_converged = min(DB['fitness0'][ind], pareto_converged)

        if DB['$marker_hifi'][ind] and DB['$marker_valid'][ind]:
            pareto_valid = min(DB['fitness0'][ind], pareto_valid)

        if DB['$marker_hifi'][ind] and DB['$marker_valid_fitness'][ind]:
            pareto_valid_fitness = min(DB['fitness0'][ind], pareto_valid_fitness)

        if DB['$marker_hifi'][ind] and DB['$marker_valid_flowparam'][ind]:
            pareto_valid_flowparam = min(DB['fitness0'][ind], pareto_valid_flowparam)

        DB['paretoConverged'][ind] = -pareto_converged
        DB['paretoValid'][ind] = -pareto_valid
        DB['paretoValidFitness'][ind] = -pareto_valid_fitness
        DB['paretoValidFlowparam'][ind] = -pareto_valid_flowparam

    for ind, position in enumerate(DB['$order']):
        if position<0:   continue
        if not DB['$marker_converged'][position]: continue
        
        pareto_converged = DB['paretoConverged'][position]
        pareto_valid = DB['paretoValid'][position]
        pareto_valid_fitness = DB['paretoValidFitness'][position]
        pareto_valid_flowparam = DB['paretoValidFlowparam'][position]
        avepareto_converged = pareto_converged
        avepareto_valid = pareto_valid
        avepareto_valid_fitness = pareto_valid_fitness
        avepareto_valid_flowparam = pareto_valid_flowparam

        ave = 0
        for i in range(0, ind):
            ind0 = DB['$order'][ind - i]
            if ave > sample: break            
            if ind0 < 0 or DB['$marker_converged'][ind0]==0: continue

            avepareto_converged = (avepareto_converged * ave + DB['paretoConverged'][ind0]) / (ave + 1.0)
            avepareto_valid = (avepareto_valid * ave + DB['paretoValid'][ind0]) / (ave + 1.0)
            avepareto_valid_fitness = (avepareto_valid_fitness * ave + DB['paretoValidFitness'][ind0]) / (ave + 1.0)
            avepareto_valid_flowparam = (avepareto_valid_flowparam * ave + DB['paretoValidFlowparam'][ind0]) / (ave + 1.0)
            ave += 1

        if pareto_converged == DB['paretoConverged'][DB['$order'][0]]:
            DB['residualConverged'][position] = 1.0
        else:
            DB['residualConverged'][position] = 1.0 - avepareto_converged / max(1e-10, pareto_converged)

        if pareto_valid == DB['paretoValid'][DB['$order'][0]]:
            DB['residualValid'][position] = 1.0
        else:
            DB['residualValid'][position] = 1.0 - avepareto_valid / max(1e-10, pareto_valid)

        if pareto_valid_fitness == DB['paretoValidFitness'][DB['$order'][0]]:
            DB['residualValidFitness'][position] = 1.0
        else:
            DB['residualValidFitness'][position] = 1.0 - avepareto_valid_fitness / max(1e-10, pareto_valid_fitness)

        if pareto_valid_flowparam == DB['paretoValidFlowparam'][DB['$order'][0]]:
            DB['residualValidFlowparam'][position] = 1.0
        else:
            DB['residualValidFlowparam'][position] = 1.0-avepareto_valid_flowparam/max(1e-10, pareto_valid_flowparam)



def aoDatabaseConvergenceReference(DB, DB_REF):

    DB['$header'] += ['maxConverged', 'maxValid', 'maxValidFitness', 'maxValidFlowparam',
                      'maxrefConverged', 'maxrefValid', 'maxrefValidFitness', 'maxrefValidFlowparam']

    maxConverged = max(DB['paretoConverged'])
    maxValid = max(DB['paretoValid'])
    maxValidFitness = max(DB['paretoValidFitness'])
    maxValidFlowparam = max(DB['paretoValidFlowparam'])
    
    maxrefConverged = max(DB_REF['paretoConverged'])
    maxrefValid = max(DB_REF['paretoValid'])
    maxrefValidFitness = max(DB_REF['paretoValidFitness'])
    maxrefValidFlowparam = max(DB_REF['paretoValidFlowparam'])
    
    DB['maxConverged'] = [1.0-x/maxConverged for x in DB['paretoConverged']]
    DB['maxValid'] = [1.0-x/maxValid for x in DB['paretoValid']]
    DB['maxValidFitness'] = [1.0-x/maxValidFitness for x in DB['paretoValidFitness']]
    DB['maxValidFlowparam'] = [1.0-x/maxValidFlowparam for x in DB['paretoValidFlowparam']]

    DB['maxrefConverged'] = [x-maxrefConverged for x in DB['paretoConverged']]
    DB['maxrefValid'] = [x-maxrefValid for x in DB['paretoValid']]
    DB['maxrefValidFitness'] = [x-maxrefValidFitness for x in DB['paretoValidFitness']]
    DB['maxrefValidFlowparam'] = [x-maxrefValidFlowparam for x in DB['paretoValidFlowparam']]
                                                                                         
#######################################################################################################################

def aoDatabaseKriging(aodir, DB):
    param = []
    data = []
 
    if os.path.isdir(aodir + '/AutoOptiProcessChain'):
        aodir = os.path.abspath(aodir + '/AutoOptiProcessChain')
    else:
        aodir = os.path.abspath(aodir)

    for key in DB['$header']:
        if 'scaled_%s' % key in DB['$header']:
            param.append(key)
            data.append([])
            for ind, marker in enumerate(DB['$marker_database']):
                if marker:
                    data[-1].append(DB[key][ind])
    data = zip(*data)
    with open('kriging2.input', 'w') as fo:        
        for _param in param:
            fo.write('%s\t' % _param)
        fo.write('\n')
        for _data in data:        
            fo.write('fit:\t0\tvariables:\t%s\n' % '\t'.join(['%g' % x for x in _data]))
    for item in sorted(os.listdir('%s/master/src' % os.path.abspath(aodir))):
        for ind, fitness in enumerate(DB['$fitness']):
            if re.match('^kriging0_flow_%d$' % fitness, item):
                hifi    = 'fitness%d_kriging_hifi' % ind
                lofi    = 'fitness%d_kriging_lofi' % ind
                DB['$header'] += [hifi, lofi]
                DB[hifi] = []
                DB[lofi] = []           
#                if sum(DB['$marker_lofi'])==0:
                if True:
                    DB[lofi] = [0] * sum(DB['$marker_database'])
                    DB[hifi] = [0] * sum(DB['$marker_database'])                    
                elif platform.system() == 'Windows':
                    pass
                elif platform.system() == 'Linux':
                    os.system('sh %s/master/src/kriging2.exe --predict --krigingfile=%s/master/src/%s --config=kriging2.input --outfile=%s_hifi.output  --predictionFidelity=0 >/dev/null 2>&1' % (os.path.abspath(aodir), os.path.abspath(aodir), item, item))
                    os.system('sh %s/master/src/kriging2.exe --predict --krigingfile=%s/master/src/%s --config=kriging2.input --outfile=%s_lofi.output  --predictionFidelity=1 >/dev/null 2>&1' % (os.path.abspath(aodir), os.path.abspath(aodir), item, item))
                    
                    with open('%s_hifi.output' % item, 'r') as fi:
                        for line in fi.readlines():
                            if re.match('\d*\.\d*', line):
                                DB[hifi].append(float(re.split('\s', line)[1]))   
                                 
                    with open('%s_lofi.output' % item, 'r') as fi:
                        for line in fi.readlines():
                            if re.match('\d*\.\d*', line):
                                DB[lofi].append(float(re.split('\s', line)[1]))  
#                    os.remove('%s_hifi.output' % item)
#                    os.remove('%s_lofi.output' % item)
    os.remove('kriging2.input')
                              
#######################################################################################################################

def aoExportStatistics(aodir, DB, DB_REF, tolerance=1e-5, archive=False):
    print('%-45s' % '    Exporting statistics...', end='')

    with open('statistics_[%s].txt' % aodir, 'w') as fo:
        stat = {"No convergence check": ['memberNumber', -1e10],
                "Check convergence (Inbuilt)": ['residualConverged', tolerance],
                "Check convergence (FlowparamROI)": ['residualValidFlowparam', tolerance],
                "Check convergence (FlowparamROI+FitnessROI)": ['residualValid', tolerance],
                }
        for key in sorted(stat.keys()):
            restriction = stat[key][0]
            eps = stat[key][1]
            zone = ['$marker_converged', '$marker_valid', '$marker_hifi', '$marker_lofi']
            header = ['Converged', 'Valid', 'HiFi', 'LoFi']
            fo.write('-' * 100 + '\n')
            fo.write('%-45s\t%15s\t%15s\t%15s\n' % (key, 'Time', 'Number', 'Percentage'))
            fo.write('-' * 100 + '\n')
            for _zone, _header in zip(zone, header):
                time_zone = 0
                time_database = 0
                number_zone = 0
                number_database = 0
                for ind in DB['$order']:
                    if ind<0:   continue
                    if DB[_zone][ind] and DB[restriction][ind] > eps:
                        time_zone += DB['memberTime'][ind]
                        number_zone += 1
                    if DB['$marker_database'][ind] and DB[restriction][ind] > eps:
                        time_database += DB['memberTime'][ind]
                        number_database += 1

                fo.write('%-45s\t' % _header)
                fo.write('%15s\t' % ('%dh/%dh' % (time_zone / 3600, time_database / 3600)))
                fo.write('%15s\t' % ('%d/%d' % (number_zone, number_database)))
                fo.write('%14.2f%%\n' % (number_zone / float(number_database) * 100.0))
            fo.write('-' * 100 + '\n\n')

        fo.write('-' * 100 + '\n')
        fo.write('%45s\t%15s\t%15s\t%15s\n' % (' ', 'Time Converged', 'Time HiFi', 'Time LoFi'))
        fo.write('-' * 100 + '\n')
        
        total_time_hifi = 0.0
        total_time_lofi = 0.0
        for key in sorted(DB.keys()):
            if re.match('timeOfProcess.+', key):
                time_converged = sum([time * marker for time, marker in zip(DB[key], DB['$marker_converged'])])
                time_hifi = sum([time * marker for time, marker in zip(DB[key], DB['$marker_hifi'])])
                time_lofi = sum([time * marker for time, marker in zip(DB[key], DB['$marker_lofi'])])
                time_converged /= max(1, sum(DB['$marker_converged']))
                time_hifi /= max(1, sum(DB['$marker_hifi']))
                time_lofi /= max(1, sum(DB['$marker_lofi']))
                total_time_hifi += time_hifi
                total_time_lofi += time_lofi
                if time_converged < 2: continue
                fo.write('%-45s\t%15d\t%15d\t%15d\n' % ('ave(%s)' % key, time_converged, time_hifi, time_lofi))

        fo.write('-' * 100 + '\n')
        fo.write('Cost ratio (CR) = %15f\n' % (total_time_lofi/max(1e-10, total_time_hifi)))
        fo.write('Replacement ratio (FR=N_CONVERGED*CR) = %15f\n' % (sum(DB['$marker_converged'])*total_time_lofi/max(1e-10, total_time_hifi)))
        for key in sorted(DB.keys()):
            if re.match('^fitness\d$', key):
                data    = []
                data.append(DB['%s_kriging_hifi' % key])
                data.append(DB['%s_kriging_lofi' % key])
                pearson = list(np.corrcoef(data))
                fo.write('Correlation between LoFi and HiFi = %.15f %.15f\n' % (pearson[0][1], pearson[0][1]**2))

    with open('convergence_[%s].dat' % aodir, 'w') as fo:
        tolerance = [3.0e-3, 2.8e-3, 2.6e-3, 2.4e-3, 2.2e-3, 2.0e-3, 1.8e-3, 1.6e-3, 1.4e-3, 1.2e-3, 1.0e-3, 8e-4, 6e-4, 4e-4, 2e-4, 1e-4]
        index = 0
        ind0 = -1
        n_converged = 0
        t_converged = 0
        n_hifi = 0
        t_hifi = 0
        n_lofi = 0
        t_lofi = 0
        fo.write('%15s, %15s, %15s, %15s, %15s, %15s, %15s\n' % ('#Tolerance', '#Time_Converged', '#Number_Converged', '#Time_HiFi', '#Number_HiFi', '#Time_LoFi', '#Number_LoFi'))
        for ind in DB['$order']:
            if ind<0:   continue
            if not DB['$marker_converged'][ind]: continue
            if ind0<0:  ind0 = ind
            if index>=len(tolerance): break
  
            if DB['$marker_converged'][ind]:
                n_converged += 1
                t_converged += DB['memberTime'][ind]/3600.0
            
            if DB['$marker_hifi'][ind]:
                n_hifi += 1
                t_hifi += DB['memberTime'][ind]/3600.0
                
            if DB['$marker_lofi'][ind]:
                n_lofi += 1
                t_lofi += DB['memberTime'][ind]/3600.0
            
            if abs(DB['maxValidFlowparam'][ind]) < tolerance[index]:
#                tolerance1 = DB['maxValidFlowparam'][ind0]
#                tolerance2 = DB['maxValidFlowparam'][ind]
#                time1 = DB['TIME'][ind0]
#                time2 = DB['TIME'][ind]
#                time_db1 = DB['TIME_DB'][ind0]
#                time_db2 = DB['TIME_DB'][ind]
#                time = ((tolerance[index]-tolerance1)*time2+(tolerance2-tolerance[index])*time1)/(tolerance2-tolerance1)
#                time_db = ((tolerance[index]-tolerance1)*time_db2+(tolerance2-tolerance[index])*time_db1)/(tolerance2-tolerance1)
                time = DB['TIME'][ind]
                time_db = DB['TIME_DB'][ind]
                fo.write('%15g, %15g, %15g, %15g, %15g, %15g, %15g\n' % (tolerance[index], t_converged, n_converged, t_hifi, n_hifi, t_lofi, n_lofi))
                index+=1
            ind0 = ind

    with open('convergence_ref_[%s].dat' % aodir, 'w') as fo:
        tolerance = [3.0e-3, 2.8e-3, 2.6e-3, 2.4e-3, 2.2e-3, 2.0e-3, 1.8e-3, 1.6e-3, 1.4e-3, 1.2e-3, 1.0e-3, 8e-4, 6e-4, 4e-4, 2e-4, 1e-4]
        index = 0
        ind0 = -1
        n_converged = 0
        t_converged = 0
        n_hifi = 0
        t_hifi = 0
        n_lofi = 0
        t_lofi = 0
        fo.write('%15s, %15s, %15s, %15s, %15s, %15s, %15s\n' % ('#Tolerance', '#Time_Converged', '#Number_Converged', '#Time_HiFi', '#Number_HiFi', '#Time_LoFi', '#Number_LoFi'))
        for ind in DB['$order']:
            if ind<0:   continue
            if not DB['$marker_converged'][ind]: continue
            if ind0<0:  ind0 = ind
            if index>=len(tolerance): break
  
            if DB['$marker_converged'][ind]:
                n_converged += 1
                t_converged += DB['memberTime'][ind]/3600.0
            
            if DB['$marker_hifi'][ind]:
                n_hifi += 1
                t_hifi += DB['memberTime'][ind]/3600.0
                
            if DB['$marker_lofi'][ind]:
                n_lofi += 1
                t_lofi += DB['memberTime'][ind]/3600.0
            
            if abs(DB['maxrefValidFlowparam'][ind]) < tolerance[index]:
#                tolerance1 = DB['maxrefValidFlowparam'][ind0]
#                tolerance2 = DB['maxrefValidFlowparam'][ind]
#                time1 = DB['TIME'][ind0]
#                time2 = DB['TIME'][ind]
#                time_db1 = DB['TIME_DB'][ind0]
#                time_db2 = DB['TIME_DB'][ind]
#                time = ((tolerance[index]-tolerance1)*time2+(tolerance2-tolerance[index])*time1)/(tolerance2-tolerance1)
#                time_db = ((tolerance[index]-tolerance1)*time_db2+(tolerance2-tolerance[index])*time_db1)/(tolerance2-tolerance1)
                time = DB['TIME'][ind]
                time_db = DB['TIME_DB'][ind]
                fo.write('%15g, %15g, %15g, %15g, %15g, %15g, %15g\n' % (tolerance[index], t_converged, n_converged, t_hifi, n_hifi, t_lofi, n_lofi))
                index+=1
            ind0 = ind
        while index<len(tolerance):
            fo.write('%15g, %15g, %15g, %15g, %15g, %15g, %15g\n' % (tolerance[index], float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')))
            index+=1
        

    if archive:
        with contextlib.closing(zipfile.ZipFile('%s.zip' % aodir, "a", zipfile.ZIP_DEFLATED)) as fo:
            fo.write(os.path.abspath('statistics_[%s].txt' % aodir), arcname='statistics_[%s].txt' % aodir)
            fo.write(os.path.abspath('convergence_[%s].dat' % aodir), arcname='convergence_[%s].dat' % aodir)
            fo.write(os.path.abspath('convergence_ref_[%s].dat' % aodir), arcname='convergence_ref_[%s].dat' % aodir)
        os.remove('statistics_[%s].txt' % aodir)
        os.remove('convergence_[%s].dat' % aodir)
        os.remove('convergence_ref_[%s].dat' % aodir)

    print(' Done')


def aoExportTecplot(aodir, DB, archive=False):
    print('%-45s' % '    Exporting database to Tecplot...', end='')

    with open('%s.dat' % aodir, 'w') as fo:
        fo.write('variables = %s\n' % (',\n'.join(DB['$header'])))

        if sum(DB['$marker_lofi']) == 0:
            zone = ['Database', 'Converged', 'Valid', 'Pareto']
            marker = ['$marker_database', '$marker_converged', '$marker_valid', '$marker_pareto']
        else:
            zone = ['Database', 'Converged', 'Valid', 'Pareto', 'HighFidelity', 'LowFidelity']
            marker = ['$marker_database', '$marker_converged', '$marker_valid', '$marker_pareto', '$marker_hifi',
                      '$marker_lofi']

        for _zone, _marker in zip(zone, marker):
            fo.write('zone T="%s" DATAPACKING=BLOCK I=%d\n' % (_zone, sum(DB[_marker])))
            for key in DB['$header']:
                count = 0
                for value, condition in zip(DB[key], DB[_marker]):
                    value = 1e10 if value==float('inf') else value
                    value = -1e10 if value==float('-inf') else value
                    if condition:
                        fo.write('%15g %s' % (value, '\n' if (count + 1) % 10 == 0 else ''))
                        count += 1
                fo.write('\n\n')

    if platform.system() == 'Windows':
        os.system('preplot %s.dat >nul' % aodir)
    elif platform.system() == 'Linux':
        os.system('preplot %s.dat >/dev/null 2>&1' % aodir)
    os.remove('%s.dat' % aodir)

    if archive:
        with contextlib.closing(zipfile.ZipFile('%s.zip' % aodir, "a", zipfile.ZIP_DEFLATED)) as fo:
            fo.write(os.path.abspath('%s.plt' % aodir), arcname='%s.plt' % aodir)
        os.remove('%s.plt' % aodir)

    print(' Done')


def aoExportVeusz(aodir, DB, archive=False):
    print('%-45s' % '    Exporting database to Veusz...', end='')

    if sum(DB['$marker_lofi']) == 0:
        zone = ['database', 'converged', 'valid', 'pareto']
        marker = ['$marker_database', '$marker_converged', '$marker_valid', '$marker_pareto']
    else:
        zone = ['database', 'converged', 'valid', 'pareto', 'hifi', 'lofi']
        marker = ['$marker_database', '$marker_converged', '$marker_valid', '$marker_pareto', '$marker_hifi',
                  '$marker_lofi']

    for _zone, _marker in zip(zone, marker):
        with open('%s_[%s].dat' % (_zone, aodir), 'w') as fo:
            header_veusz = []
            for _header in DB['$header']:
                if not _header in DB['$header_flowparam']:
                    header_veusz.append(_header)
#            fo.write('#%s\n' % (', #'.join(DB['$header'])))
            fo.write('#%s\n' % (', #'.join(header_veusz)))
            for ind, condition in enumerate(DB[_marker]):
                if condition:
#                    for key in DB['$header']:
                    for key in header_veusz:
                        fo.write('%15g, ' % DB[key][ind])
                    fo.write("\n")
        if archive:
            with contextlib.closing(zipfile.ZipFile('%s.zip' % aodir, "a", zipfile.ZIP_DEFLATED)) as fo:
                fo.write(os.path.abspath('%s_[%s].dat' % (_zone, aodir)), arcname='%s_[%s].dat' % (_zone, aodir))
            os.remove('%s_[%s].dat' % (_zone, aodir))

    print(' Done')


def aoExportPearson(aodir, DB, archive=False):
    print('%-45s' % '    Exporting Pearson coefficient...', end='')
    fitness = []
    data = []
    param = []
    value = []
    member = []
    
    for ind, marker in enumerate(DB['$marker_pareto']):
        if marker:
            member.append(DB['memberNumber'][ind])

    for key in sorted(DB.keys()):
        if re.match('^fitness\d+$', key):
            fitness.append(key)
            param.append(key)
            data.append([])
            value.append([])
            for ind, marker in enumerate(DB['$marker_valid']):
                if marker:
                    data[-1].append(DB[key][ind])
            for ind, marker in enumerate(DB['$marker_pareto']):
                if marker:
                    value[-1].append(DB[key][ind])

        if re.match('scaled_.+', key):
            param.append(re.sub('scaled_', '', key))
            data.append([])
            value.append([])
            for ind, marker in enumerate(DB['$marker_valid']):
                if marker:
                    data[-1].append(DB[key][ind])
            for ind, marker in enumerate(DB['$marker_pareto']):
                if marker:
                    value[-1].append(DB[key][ind])

    pearson = list(np.corrcoef(data))

    with open('pearson_[%s].dat' % aodir, 'w') as fo:
        fo.write('descriptor param')
        for _fitness in fitness:
            fo.write(' %s %s_abs %s_sqr' % (_fitness, _fitness, _fitness))
        for _member in member:
            fo.write(' value%d'%_member)
        fo.write('\n')
        
        
        
#        for i, _param, _value in zip(range(len(fitness), len(pearson)), param, value):
        for i, _param, _value in zip(range(len(pearson)), param, value):
            fo.write('"%s"' % _param)
            for j in range(len(fitness)):
                fo.write(' %g' % pearson[i][j])
                fo.write(' %g' % abs(pearson[i][j]))
                fo.write(' %g' % pearson[i][j]**2)
            for v in _value:
                fo.write(' %g' % v)
            fo.write("\n")

    if archive:
        with contextlib.closing(zipfile.ZipFile('%s.zip' % aodir, "a", zipfile.ZIP_DEFLATED)) as fo:
            fo.write(os.path.abspath('pearson_[%s].dat' % aodir), arcname='pearson_[%s].dat' % aodir)
        os.remove('pearson_[%s].dat' % aodir)

    print(' Done')


def aoInitializeArchive(aodir):
    print('%-45s' % '    Initializing output archive...', end='')
    contextlib.closing(zipfile.ZipFile('%s.zip' % aodir, "w", zipfile.ZIP_DEFLATED))
    print(' Done')


def aoExportDatabase(aodir):
    print('%-45s' % '    Exporting database...', end='')

    if aoDatabaseCheck(aodir):
        with contextlib.closing(zipfile.ZipFile('%s.zip' % aodir, "a", zipfile.ZIP_DEFLATED)) as fo:
            aofiles = ['Output/database.csv', 'Input/ResultNames',
                       'Input/genData.input', 'Input/OptimizationParameters']
            if os.path.isdir(aodir + '/AutoOptiProcessChain'):
                aodir += '/AutoOptiProcessChain'

            for item in sorted(os.listdir('%s/master/src' % os.path.abspath(aodir))):
                if re.match('^kriging0_flow_\d+$', item):
                    aofiles += ['src/%s' % item]

            for item in aofiles:
                fo.write('%s/master/%s' % (os.path.abspath(aodir), item), arcname='master/%s' % item)



    print(' Done')

########################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoOpti database post-processing tool')

    parser.add_argument('-a', '--archive', default=False, action="store_true",
                        help='Compress all script output to archive')

    parser.add_argument('-c', '--convergence', dest='convergence', type=float, nargs=2,
                        help='Specify averaging sample and tolerance for convergence criterion (default 100, 1e-5)')

    parser.add_argument('-f', '--filter', dest='filter', type=str, nargs=1,
                        help='Specify string to filter the processed folders')

    parser.add_argument('-r', '--reference', dest='reference', type=str, nargs=1,
                        help='Specify path to the reference optimization')

    parser.add_argument('-d', '--database', default=False, action="store_true",
                        help='Export AutoOpti database to compressed file')

    parser.add_argument('-p', '--pearson', default=False, action="store_true",
                        help='Export correlations between fitness function and input parameters')

    parser.add_argument('-s', '--statistics', default=False, action="store_true",
                        help='Print statistics on the AutoOpti database')

    parser.add_argument('-t', '--tecplot', default=False, action="store_true",
                        help='Export the AutoOpti database to TecPlot')

    parser.add_argument('-v', '--veusz', default=False, action="store_true",
                        help='Export the AutoOpti database to Veusz')

    parser.set_defaults(convergence=[400, 1e-5], filter=['.*'], reference=[None])
    args = parser.parse_args()
    workdir = os.getcwd()

    for aodir in ([workdir] + sorted(os.listdir(workdir))):
        if not re.match(args.filter[0], aodir):
            pass
        elif not aoDatabaseCheck(aodir):
            pass
        else:
            if args.tecplot or args.veusz or args.pearson or args.statistics:
                print('%-60s' % aodir, end='')
                DB, HEADER = aoImportDatabase(aodir)
                if len(DB['$marker_database'])==0:
                   print(' 0/0')
                   continue
                aoDatabaseConvergence(DB, int(args.convergence[0]))
#                aoDatabaseKriging(aodir, DB)
                print(' %d/%d' % (sum(DB['$marker_converged']), sum(DB['$marker_database'])))

                if args.reference[0] is not None:
                    print('%-45s' % '    Importing reference database...', end='')
                    DB_REF, HEADER_REF = aoImportDatabase(args.reference[0])
                    aoDatabaseConvergence(DB_REF, int(args.convergence[0]))
                    aoDatabaseConvergenceReference(DB, DB_REF)
                    print(' Done')
                    print('%-45s' % ('        %s' % args.reference[0]), end='')
                    print(' %d/%d' % (sum(DB_REF['$marker_converged']), sum(DB_REF['$marker_database'])))
                else:
                    aoDatabaseConvergenceReference(DB, DB)
            else:
                print('%-60s' % aodir)

            aodir += '/database' if aodir == os.getcwd() else ''

            if args.archive or args.database:
                aoInitializeArchive(aodir)
            if args.database:
                aoExportDatabase(aodir)
            if args.tecplot:
                aoExportTecplot(aodir, DB, args.archive)
            if args.veusz:
                aoExportVeusz(aodir, DB, args.archive)
            if args.pearson:
                aoExportPearson(aodir, DB, args.archive)
            if args.statistics:
                if args.reference[0] is not None:
                    aoExportStatistics(aodir, DB, DB_REF, float(args.convergence[1]), args.archive)
		else:
                    aoExportStatistics(aodir, DB, DB, float(args.convergence[1]), args.archive)

print('Done')
########################################################################################################################
