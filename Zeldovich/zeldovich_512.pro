;
;
;
;
;
;


head = read_snap_head('snapdir_000/snap_000.GAD')
omega_m = head.OMEGA0
omega_l = head.OMEGALAMBDA
H0      = head.HUBBLEPARAM
box     = head.boxsize

scales  = [500.0, 1000.0, 2000.0]
k_fourier = box / scales

ng = 512L

psi_x0 = read_fvolume('Data/PSI_X.fvol')
psi_y0 = read_fvolume('Data/PSI_Y.fvol')
psi_z0 = read_fvolume('Data/PSI_Z.fvol')

;------------------------------------------
;  Loop over scales
;------------------------------------------
FOR w=0L, 3 DO BEGIN

;--- Create k-sharp filter
mask = sharp_k(ng, k_fourier[w], 0)
;--- Make composite field in Fourier:
fft_par = fft(psi_x0,  -1)
psi_x = float(fft(fft_par*mask, 1))
fft_par = fft(psi_y0,  -1)
psi_y = float(fft(fft_par*mask, 1))
fft_par = fft(psi_z0,  -1)
psi_z = float(fft(fft_par*mask, 1))


;--- Lagrangian coordinates
qx = fltarr(ng,ng,ng)
qy = fltarr(ng,ng,ng)
qz = fltarr(ng,ng,ng)
center = box/ng/2.0
cont=0L
FOR k=0L, ng-1 DO BEGIN
   FOR j=0L, ng-1 DO BEGIN
      FOR i=0L, ng-1 DO BEGIN
      
         qx[i,j,k] = (i/float(ng))*box + center
         qy[i,j,k] = (j/float(ng))*box + center
         qz[i,j,k] = (k/float(ng))*box + center

         cont++
      ENDFOR
   ENDFOR
   print, k
ENDFOR


;--- Define new redshift
zf = 46.77
af = 1.0/(zf+1.0)

;--- Hubbe
H_z = H0*sqrt(omega_m/af^3 + omega_l)

;--- Matter content
omega_m_z = (omega_m/af^3)/(omega_m/af^3 + omega_l)

;--- Approximation to d log D / d log a
f_z = omega_m_z^(5.0/9.0)

;--- Grwth function
D_z = growth(omega_m, omega_L, zf)/growth(omega_m, omega_L, 0.0)

;--- Zeldovich
px = qx + psi_x * D_z
py = qy + psi_y * D_z
pz = qz + psi_z * D_z

vx = af * H_z * f_z * D_z * psi_x
vy = af * H_z * f_z * D_z * psi_y
vz = af * H_z * f_z * D_z * psi_z


;---- Linear positions
px = reform(px,ng^3)
py = reform(py,ng^3)
pz = reform(pz,ng^3)
make_periodic, box, px,py,pz

vx = reform(vx,ng^3)
vy = reform(vy,ng^3)
vz = reform(vz,ng^3)
id = lindgen(ng^3)

head = read_snap_head('snapdir_000/snap_000.GAD')
head.redshift      = zf
head.time          = af
head.massarr[1]    = head.massarr[1] * (double(455)^3 / double(ng)^3)
head.npart[1]      = ng^3
head.nparttotal[1] = ng^3
head.numfiles      = 1

write_custom_gad, head, px,py,pz, vx,vy,vz, id, 'IC/IC_512-SCL'+format_number(w,2)+'.GAD'

ENDFOR

;wsize, 900
;lin = lindgen(ng^2)
;plot, px[lin], py[lin], psym=3,/xs,/ys,position=[0,0,1,1]


 


end
