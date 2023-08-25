import json
import pandas as pd

import os

def format_merged_cell(value1, value2):
    merged_cell = f'<div style="display: flex; flex-direction: row;"><div style="width: 50%;">{value1}</div><div style="width: 50%;">{value2}</div></div>'
    return merged_cell

def format_to_json(results):
    df = pd.DataFrame(columns=['Index'], data={'Index': ['Nucleotide','hAm','hCm','hGm','hTm','hm1A',
                                                         'hm5C','hm5U','hm6A','hm6Am','hm7G','hPsi','Atol']})
    # print(df)

    json_data = json.loads(results)
    print(json_data)

    dataWithProbabilities = json_data["POSITION_WITH_PROBABILITIES"]
    print(dataWithProbabilities)

    for data in dataWithProbabilities:
        binaryIndex = str(data['RNA_MODIFIED_INDEX']) + '-binary'
        binaryColumn = getBinaryColumn(data, data['PARENT_MODIFIED_NUCLEOTIDE'], 'BINARY_MODIFICATION_PROBABILITIES')
        df[binaryIndex] = binaryColumn
        multiIndex = str(data['RNA_MODIFIED_INDEX']) + '-multi'
        multiColumn = getColumn(data, '', 'MULTICLASS_MODIFICATION_PROBABILITIES')
        df[multiIndex] = multiColumn
        finalIndex = str(data['RNA_MODIFIED_INDEX']) + '-final'
        multiColumn = getFinalColumn(data['BINARY_MODIFICATION_PROBABILITIES'], data['MULTICLASS_MODIFICATION_PROBABILITIES'])
        df[finalIndex] = multiColumn
    return df

def getColumn(jsonObject, nucleotide, type):
    # print('jsonObject: ', jsonObject)
    probabilities = jsonObject[type]
    hAm = 0.0
    hCm = 0.0
    hGm = 0.0
    hTm = 0.0
    hm1A = 0.0
    hm5C = 0.0
    hm5U = 0.0
    hm6A = 0.0
    hm6Am = 0.0
    hm7G = 0.0
    hPsi = 0.0
    Atol = 0.0
    for p in probabilities:
        print("p: ", p)
        if "hAm" in p:
            hAm = round(float(p['hAm']),3)
        if "hCm" in p:
            hCm = round(float(p['hCm']),3)
        if "hGm" in p:
            hGm = round(float(p['hGm']),3)
        if "hTm" in p:
            hTm = round(float(p['hTm']),3)
        if "hm1A" in p:
            hm1A = round(float(p['hm1A']),3)
        if "hm5C" in p:
            hm5C = round(float(p['hm5C']),3)
        if "hm5U" in p:
            hm5U = round(float(p['hm5U']),3)
        if "hm6A" in p:
            hm6A = round(float(p['hm6A']),3)
        if "hm6Am" in p:
            hm6Am = round(float(p['hm6Am']),3)
        if "hm7G" in p:
            hm7G = round(float(p['hm7G']),3)
        if "hPsi" in p:
            hPsi = round(float(p['hPsi']),3)
        if "Atol" in p:
            Atol = round(float(p['Atol']),3)

    if hAm == 0.0:
        hAm = ''
    if hCm == 0.0:
        hCm = ''
    if hGm == 0.0:
        hGm = ''
    if hTm == 0.0:
        hTm = ''
    if hm1A == 0.0:
        hm1A = ''
    if hm5C == 0.0:
        hm5C = ''
    if hm5U == 0.0:
        hm5U = ''
    if hm6A == 0.0:
        hm6A = ''
    if hm6Am == 0.0:
        hm6Am = ''
    if hm7G == 0.0:
        hm7G = ''
    if hPsi == 0.0:
        hPsi = ''
    if Atol == 0.0:
        Atol = ''
    column = [nucleotide, hAm, hCm, hGm, hTm, hm1A, hm5C, hm5U, hm6A, hm6Am, hm7G, hPsi, Atol]

    print("column: ", column)
    return column

def getBinaryColumn(jsonObject, nucleotide, type):
    # print('jsonObject: ', jsonObject)
    probabilities = jsonObject[type]
    hAm = 0.0
    hCm = 0.0
    hGm = 0.0
    hTm = 0.0
    hm1A = 0.0
    hm5C = 0.0
    hm5U = 0.0
    hm6A = 0.0
    hm6Am = 0.0
    hm7G = 0.0
    hPsi = 0.0
    Atol = 0.0
    for p in probabilities:
        print("p: ", p)
        if "hAm" in p:
            hCm = round(float(p['hAm'][0]),3)
        if "hCm" in p:
            hCm = round(float(p['hCm'][0]),3)
        if "hGm" in p:
            hGm = round(float(p['hGm'][0]),3)
        if "hTm" in p:
            hTm = round(float(p['hTm'][0]),3)
        if "hm1A" in p:
            hm1A = round(float(p['hm1A'][0]),3)
        if "hm5C" in p:
            hm1A = round(float(p['hm5C'][0]),3)
        if "hm5U" in p:
            hm5U = round(float(p['hm5U'][0]),3)
        if "hm6A" in p:
            hm6A = round(float(p['hm6A'][0]),3)
        if "hm6Am" in p:
            hm6Am = round(float(p['hm6Am'][0]),3)
        if "hm7G" in p:
            hm7G = round(float(p['hm7G'][0]),3)
        if "hPsi" in p:
            hPsi = round(float(p['hPsi'][0]),3)
        if "Atol" in p:
            Atol = round(float(p['Atol'][0]),3)

    if hAm == 0.0:
        hAm = ''
    if hCm == 0.0:
        hCm = ''
    if hGm == 0.0:
        hGm = ''
    if hTm == 0.0:
        hTm = ''
    if hm1A == 0.0:
        hm1A = ''
    if hm5C == 0.0:
        hm5C = ''
    if hm5U == 0.0:
        hm5U = ''
    if hm6A == 0.0:
        hm6A = ''
    if hm6Am == 0.0:
        hm6Am = ''
    if hm7G == 0.0:
        hm7G = ''
    if hPsi == 0.0:
        hPsi = ''
    if Atol == 0.0:
        Atol = ''
    column = [nucleotide, hAm, hCm, hGm, hTm, hm1A, hm5C, hm5U, hm6A, hm6Am, hm7G, hPsi, Atol]

    print("column: ", column)
    return column

def getValue(binaryData, multiData, param):
    binary = 0.0
    multi = 0.0

    for p in binaryData:
        if param in p:
            binary = float(p[param][0])

    for p in multiData:
        if param in p:
            multi = float(p[param])

    if binary == 0.0 and multi == 0.0:
        return ''
    else:
        return round(binary * multi,3)

def getFinalColumn(binaryData, multiData):
    nucleotide = ''

    hAm = getValue(binaryData, multiData, 'hAm')
    hCm = getValue(binaryData, multiData, 'hCm')
    hGm = getValue(binaryData, multiData, 'hGm')
    hTm = getValue(binaryData, multiData, 'hTm')
    hm1A = getValue(binaryData, multiData, 'hm1A')
    hm5C = getValue(binaryData, multiData, 'hm5C')
    hm5U = getValue(binaryData, multiData, 'hm5U')
    hm6A = getValue(binaryData, multiData, 'hm6A')
    hm6Am = getValue(binaryData, multiData, 'hm6Am')
    hm7G = getValue(binaryData, multiData, 'hm7G')
    hPsi = getValue(binaryData, multiData, 'hPsi')
    Atol = getValue(binaryData, multiData, 'Atol')

    column = [nucleotide, hAm, hCm, hGm, hTm, hm1A, hm5C, hm5U, hm6A, hm6Am, hm7G, hPsi, Atol]

    # print("column: ", column)
    return column

def save_json_to_excel(data, filename):
    df = pd.DataFrame(data)
    transposed_df = df.T
    downloads_path = os.path.expanduser("~") + "/Downloads/" + filename
    transposed_df.reset_index().to_excel(downloads_path, index=False)
    return downloads_path
