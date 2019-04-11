;
;
;
;
;
;

head = read_snap_head('../snapdir_000/snap_000.GAD')

omega_m = head.OMEGA0
omega_l = head.OMEGALAMBDA
H0      = head.HUBBLEPARAM
z       = head.REDSHIFT
box     = head.boxsize

a       = 1.0/(z+1.0)


FLAG_READ = 0
IF FLAG_READ NE 0 THEN BEGIN

;--- Read velocities from DTFE interpolation
vx = read_dvolume('snap_000.vx.dvol')
vy = read_dvolume('snap_000.vy.dvol')
vz = read_dvolume('snap_000.vz.dvol')



;--- Hubble constant
H_z = H0*sqrt(omega_m/a^3 + omega_l)

;--- 
omega_m_z = (omega_m/a^3)/(omega_m/a^3 + omega_l)

;--- Appproximation of d log D / d log a
f_z = omega_m_z^(5.0/9.0)

;--- Growth function
D_z = growth(omega_m, omega_L, z)/growth(omega_m, omega_L, 0.0)


;--- Displacement fields
psi_x = vx / a / H_z / f_z / D_z 
psi_y = vy / a / H_z / f_z / D_z
psi_z = vz / a / H_z / f_z / D_z


ng = 512L
write_fvolume, psi_x, ng,ng,ng,ng, 0,0,0, 'PSI_X.fvol'
write_fvolume, psi_y, ng,ng,ng,ng, 0,0,0, 'PSI_Y.fvol'
write_fvolume, psi_z, ng,ng,ng,ng, 0,0,0, 'PSI_Z.fvol'


ENDIF

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
af = 50.0
zf = 1.0/af - 1

;--- Hubbe
H_z = H0*sqrt(omega_m/a^3 + omega_l)

;--- Matter content
omega_m_z = (omega_m/af^3)/(omega_m/af^3 + omega_l)

;--- Approximation to d log D / d log a
f_z = omega_m_z^(5.0/9.0)

;--- Grwth function
D_z = growth(omega_m, omega_L, zf)/growth(omega_m, omega_L, 0.0)


;--- Zeldovich again
px = qx + psi_x * D_z
py = qx + psi_y * D_z
pz = qx + psi_z * D_z


vx = af * H_z * f_z * D_z * psi_x




end

