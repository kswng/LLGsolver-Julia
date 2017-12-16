# https://github.com/kswng/Landau-Lifshits-Gilbert

Lx = 10  # system
Ly = 10  # size

step = 0.01  # timestep

alpha = 0.01 # Gilbert damping const.

maxtime = 100000


J = -1  # exchange coupling

Bfield = 0.1 * [0; 0; 1] # zeeman field






function normalize_system(spin_x,spin_y,spin_z)

    norm = sqrt(spin_x[2:Ly+1,2:Lx+1].^2 + spin_y[2:Ly+1,2:Lx+1].^2 + spin_z[2:Ly+1,2:Lx+1].^2)
    spin_x[2:Ly+1,2:Lx+1]./=norm
    spin_y[2:Ly+1,2:Lx+1]./=norm
    spin_z[2:Ly+1,2:Lx+1]./=norm

    return spin_x, spin_y, spin_z
end



function Heisenberg(x, y, spin_x, spin_y, spin_z)

    vec0 = [spin_x[y,x] ; spin_y[y,x] ; spin_z[y,x]]
    vecr = [spin_x[y,x+1] ; spin_y[y,x+1] ; spin_z[y,x+1]]
    vecl = [spin_x[y,x-1] ; spin_y[y,x-1] ; spin_z[y,x-1]]
    vecu = [spin_x[y-1,x] ; spin_y[y-1,x] ; spin_z[y-1,x]]
    vecd = [spin_x[y+1,x] ; spin_y[y+1,x] ; spin_z[y+1,x]]

    return  J * (vecr + vecl + vecu + vecd)

end


function Zeeman(x, y, spin_x, spin_y, spin_z)

    return -Bfield

end



function Kfunc(spin_x, spin_y, spin_z, kx, ky, kz, time)

    kx = kx .*box
    ky = ky .*box
    kz = kz .*box

    sx_new = spin_x + kx
    sy_new = spin_y + ky
    sz_new = spin_z + kx


    eff_field = zeros(Ly +2, Lx + 2,3)

    for x in 2:Lx+1
        for y in 2:Ly+1
            eff_field[y,x,:]  = Heisenberg(x, y, sx_new, sy_new, sz_new) +Zeeman(x, y,sx_new,sy_new,sz_new)
        end
    end


    output_vect = zeros(Ly +2, Lx + 2, 3)


    for x in 2:Lx+1
        for y in 2:Ly+1
            spinvec = [sx_new[y,x]; sy_new[y,x]; sz_new[y,x]]
            output_vect[y,x,:]  = -1/(1 + alpha^2) * (cross(spinvec,eff_field[y,x,:]- alpha*cross(spinvec,eff_field[y,x,:])))
        end
    end

    return output_vect[:,:,1],output_vect[:,:,2],output_vect[:,:,3]


end



function update_spin(spin_x,spin_y,spin_z,time)  # 4th order Runge-Kutta

    sx = copy(spin_x)

    k1_sx, k1_sy, k1_sz = Kfunc(spin_x,spin_y,spin_z,zeros(Ly+2,Lx+2),zeros(Ly+2,Lx+2),zeros(Ly+2,Lx+2),time)


    k2_sx, k2_sy, k2_sz =Kfunc(spin_x,spin_y,spin_z,k1_sx * step/2, k1_sy* step/2, k1_sz* step/2,time +step/2)


    k3_sx, k3_sy, k3_sz  = Kfunc(spin_x,spin_y,spin_z,k2_sx * step/2, k2_sy* step/2, k2_sz* step/2,time +step/2)

    k4_sx, k4_sy, k4_sz = Kfunc(spin_x,spin_y,spin_z,k3_sx * step, k3_sy* step, k3_sz* step,time +step)

    spin_x += step*( (k1_sx + 2 *k2_sx +2 *k3_sx + k4_sx))/6
    spin_y += step*( (k1_sy + 2 *k2_sy +2 *k3_sy + k4_sy))/6
    spin_z += step*( (k1_sz + 2 *k2_sz +2 *k3_sz + k4_sz))/6

    return spin_x, spin_y, spin_z

end



### initial spin config ###

spin_x = (2*rand(Ly+2,Lx+2)-1)
spin_y = (2*rand(Ly+2,Lx+2)-1)
spin_z = (2*rand(Ly+2,Lx+2)-1)

spin_x, spin_y, spin_z = normalize_system(spin_x,spin_y,spin_z)

box = zeros(Ly+2,Lx+2)
box[2:Ly+1,2:Lx+1] = ones(Ly,Lx)




for time_ind in 0:maxtime

    time = time_ind * step


    spin_x, spin_y, spin_z = update_spin(spin_x, spin_y, spin_z, time)

    spin_x, spin_y, spin_z = normalize_system(spin_x,spin_y,spin_z)
    ## OBC
    spin_x = spin_x .*box
    spin_y = spin_y .*box
    spin_z = spin_z .*box

    println(sum(spin_z)/Lx/Ly)

end
