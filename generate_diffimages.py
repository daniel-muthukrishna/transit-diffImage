import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tess_stars2px
from tess_stars2px import tess_stars2px_function_entry

from transitDiffImage import tessDiffImage, transitCentroids
from transitDiffImage import tessprfmodel as tprf

import multiprocessing
from multiprocessing import Pool


# Load Vetting csv file with list of TIC IDs
fpath_vetting = "/pdo/users/dmuth/mnt/tess/labels/vetting-v02.csv"
vet_table = pd.read_csv(fpath_vetting, header=0, low_memory=False).set_index('Astro ID')


def process_astro_id(astro_id):
    ## Get TIC information for a given Astro ID
    star = {}
    star['id'] = vet_table['TIC ID'][astro_id]
    star['raDegrees'] = vet_table['RA'][astro_id]
    star['decDegrees'] = vet_table['Dec'][astro_id]

    planet0 = {}
    planet0['planetID'] = f"astroid{astro_id}"
    planet0['period'] = vet_table['Period'][astro_id]
    planet0['epoch'] = vet_table['Epoc'][astro_id]
    planet0['durationHours'] = vet_table['Duration'][astro_id] * 24

    ## Create directory to save images and redirect stdout and stderr to outputs.log
    dirname = f"tic-images/tic{star['id']}"
    os.makedirs(dirname, exist_ok=True)


    if os.path.exists(f'{dirname}/centroid_distance_astroid{astro_id}.txt'):
        print(f"Already done this Astro ID ({astro_id}), skipping...")
        return
    

    with open(f'{dirname}/outputs.log','wt') as outputs_f:
        sys.stdout = outputs_f
        sys.stderr = outputs_f
        print(f"Astro ID: {astro_id}\n ---------------------------------")

        ## Use TESSpoint to get which sectors TIC is observed in
        outID, outEclipLong, outEclipLat, outSec, outCam, outCcd, \
            outColPix, outRowPix, scinfo = tess_stars2px_function_entry(star['id'], 
                                                                        star['raDegrees'], 
                                                                        star['decDegrees'])
        print(outSec, outCam)

        for i in range(len(outSec)):
            print(f"Sector: {outSec[i]}\n -------------------")

            try:

                ## Use TESScut to download FFI cutout, and calculate difference image
                star['sector'] = outSec[i]
                star['cam'] = outCam[i]
                star['planetData'] = [planet0]
                star['qualityFiles'] = None 
                star['qualityFlags'] = None
                print(star)

                tdi = tessDiffImage.tessDiffImage(star, outputDir='/pdo/users/dmuth/Projects/Generate_TESS_Diff_Images/transit-diffImage/tic-images/')
                tdi.make_ffi_difference_image(thisPlanet=0)

                sectorIndex = tdi.sectorList.index(star['sector'])


                ## Load Image Data
                fname = f"{dirname}/imageData_{planet0['planetID']}_sector{star['sector']}.pickle"
                with open(fname, 'rb') as f:
                    imageData = pickle.load(f)

                # Plot difference image
                diffImageData = imageData[0]
                catalogData = imageData[1]

                fig, ax = plt.subplots(2,2,figsize=(10,10))
                tdi.draw_pix_catalog(diffImageData['diffImage'], catalogData, catalogData["extent"], ax=ax[0,0], fs=14, ss=60, filterStars=True, dMagThreshold=4, annotate=True)
                tdi.draw_pix_catalog(diffImageData['diffImage'], catalogData, catalogData["extentClose"], ax=ax[0,1], fs=14, ss=60, filterStars=True, dMagThreshold=4, annotate=True, close=True)
                tdi.draw_pix_catalog(diffImageData['meanOutTransit'], catalogData, catalogData["extent"], ax=ax[1,0], fs=14, ss=60, filterStars=True, dMagThreshold=4, annotate=True)
                tdi.draw_pix_catalog(diffImageData['meanOutTransit'], catalogData, catalogData["extentClose"], ax=ax[1,1], fs=14, ss=60, filterStars=True, dMagThreshold=4, annotate=True, close=True)
                ax[0,0].set_title('Difference Image')
                ax[0,1].set_title('Difference Image (Close-up)')
                ax[1,0].set_title('Direct Image')
                ax[1,1].set_title('Direct Image (Close-up)')

                plt.savefig(f"{dirname}/diffImage_{planet0['planetID']}_sector{star['sector']}.png")
                plt.close() 


                ## Calculate Centroids

                # Create TESS PRF object
                prf = tprf.SimpleTessPRF(shape=diffImageData["diffImage"].shape,
                                        sector = outSec[sectorIndex],
                                        camera = outCam[sectorIndex],
                                        ccd = outCcd[sectorIndex],
                                        column=catalogData["extent"][0],
                                        row=catalogData["extent"][2],
                                        # prfFileLocation = "../../tessPrfFiles/"
                )

                # Compute the centroid
                fitVector, prfFitQuality, fluxCentroid, closeDiffImage, closeExtent = transitCentroids.tess_PRF_centroid(prf, 
                                                            catalogData["extent"], 
                                                            diffImageData["diffImage"], 
                                                            catalogData)
                
                print("PRF fit quality = " + str(prfFitQuality))


                # Compute the centroid in RA and Dec
                raDec = tessDiffImage.pix_to_ra_dec(tdi.sectorList[sectorIndex], 
                                    outCam[sectorIndex], outCcd[sectorIndex], 
                                    fitVector[0], fitVector[1])
                print(raDec)
                print(tessDiffImage.pix_distance(raDec, tdi.sectorList[sectorIndex], outCam[sectorIndex], outCcd[sectorIndex],
                                        fitVector[0], fitVector[1]))

                dRa = raDec[0] - catalogData['correctedRa'][0]
                dDec = raDec[1] - catalogData['correctedDec'][0]
                centroid_distance = str(3600*np.sqrt((dRa*np.cos(catalogData['correctedDec'][0]*np.pi/180))**2 + dDec**2))
                print("distance = " + centroid_distance + " arcsec")

                
                # Show the flux-weighted and PRF-fit centroids on the difference image, along with the position of the target star (the first star in the catalog data).
                plt.imshow(closeDiffImage, cmap='jet', origin='lower', extent=closeExtent)
                plt.plot(fluxCentroid[0], fluxCentroid[1], 'w+', label = "flux-weighted centroid", zorder=200)
                plt.plot(fitVector[0], fitVector[1], 'ws', label = "PRF-fit centroid", zorder=200)
                plt.axvline(catalogData["targetColPix"][0], c='y', label = "target star")
                plt.axhline(catalogData["targetRowPix"][0], c='y')
                plt.colorbar()
                plt.legend()
                plt.title(f"Centroid Distance: {centroid_distance} arcsec")
                plt.savefig(f"{dirname}/centroid_diffImage_{planet0['planetID']}_sector{star['sector']}.png")
                plt.close()


                # Save centroid distance to file
                with open(f'{dirname}/centroid_distance_astroid{astro_id}.txt', 'a') as f_centroids:
                    f_centroids.write(f"{astro_id},{star['id']},{star['sector']},{centroid_distance}")

            except Exception as e:
                print("###########\n Error:", e, "\n############")

        

def create_centroid_distance_csv(astro_ids):
    with open('centroid_distance_astro_ids.csv', 'a') as f_combined:
        f_combined.write("Astro ID,TIC ID,Sector,Centroid Distance\n")
        for astro_id in astro_ids:
            dirname = f"tic-images/tic{vet_table['TIC ID'][astro_id]}"
            filename = f'{dirname}/centroid_distance_astroid{astro_id}.txt'
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    data = f.readline().strip().split(',')
                    f_combined.write(f"{data[0]},{data[1]},{data[2]}\n")



def quick_flux_centroid(arr, extent, constrain=True):
    xpix = np.linspace(extent[0], extent[1]-1, arr.shape[1])
    ypix = np.linspace(extent[2], extent[3]-1, arr.shape[0])
    X, Y = np.meshgrid(xpix, ypix)
    normArr = arr.copy() - np.median(arr.ravel())
    sum_f = np.sum(normArr.ravel())
    sum_x = np.sum((X*normArr).ravel())
    sum_y = np.sum((Y*normArr).ravel())
    
    xc = sum_x/sum_f
    yc = sum_y/sum_f
    
    if constrain:
        # if the centroid is outside the extent then return the center of the image
        if (xc < extent[0]) | (xc > extent[1]):
            xc = np.mean(extent[0:2])

        if (yc < extent[2]) | (yc > extent[3]):
            yc = np.mean(extent[2:])

    return [xc, yc]


def main():
    # Get list of Astro IDs
    astro_ids = vet_table.index.values  # np.arange(1, 4) #

    # Process Astro IDs in parallel
    num_processes = multiprocessing.cpu_count() - 8
    with Pool(processes=num_processes) as pool:
        pool.map(process_astro_id, astro_ids)

    # Serial processing
    # for astro_id in astro_ids:
    #     process_astro_id(astro_id)

    # Create centroid_distance_astro_ids.csv
    create_centroid_distance_csv(astro_ids)

if __name__ == "__main__":
    main()



    
