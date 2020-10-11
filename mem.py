import ehtim as eh


iccfits = "/home/ilya/eht/cc_I_001.fits"
qccfits = "/home/ilya/eht/cc_Q_001.fits"
uccfits = "/home/ilya/eht/cc_U_001.fits"
uvfits = "/home/ilya/eht/0212+735.u.2006_07_07.uvf"

obs = eh.obsdata.load_uvfits(uvfits, polrep="circ")
imicc = eh.image.load_fits(iccfits, aipscc=True)
im = eh.image.load_fits(iccfits, aipscc=False)
imi = eh.image.load_fits(iccfits, aipscc=False)
imq = eh.image.load_fits(qccfits, aipscc=False)
imu = eh.image.load_fits(uccfits, aipscc=False)
im.add_qu(imq.imarr(), imu.imarr())

# Resolution
beamparams = obs.fit_beam() # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
res = obs.res() # nominal array resolution, 1/longest baseline
print("Clean beam parameters: " , beamparams)
print("Nominal Resolution: " , res)

npix = 512
fov = 1*imi.fovx()
zbl = imicc.total_flux() # total flux
prior = im

flux = zbl
imgr = eh.imager.Imager(obs, prior, prior, flux,
                        data_term={'vis': 5}, show_updates=False,
                        reg_term={'tv': 2, "l1": 1, 'flux': 3},
                        maxit=100, ttype='nfft')
imgr.make_image_I(grads=True)

out = imgr.out_last().threshold(0.01)
imgr.init_next = out.blur_circ(0.25*res)
im = imgr.init_next
# im.add_random_pol(0.1, 3.14/4)
imgr.prior_next = im
imgr.transform_next = 'mcv'
imgr.dat_term_next = {'pvis': 1}
imgr.reg_term_next = {'hw': 1}
imgr.maxit_next = 200
imgr.make_image_P()
out = imgr.out_last()
out.display(pol="m")
