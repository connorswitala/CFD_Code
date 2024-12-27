%% This code solves the Sod shock tube problem using Steger-Warming flux vector splitting 
%% applied to a finite-volume method in Matlab. The geometry is very simple so none of the face normals,
%% face areas, or volumes need to be stored for specific cells, they all have the same value.
%% Two figures are plotted. The first is a contour plot of the entire domain for all primitive 
%% variable. The second is a line plot that takes the center line of the domain and plots all the values. 
%% This is helpful for comparing against the exact solution of a 1D Sod shock tube problem. All
%% you need to do is run this code, nothing special needs to be done. There are 10 cells in the y-direction
%% and 4000 cells in the x-direction, so it takes a relatively decent amount of time to solve. If too few 
%% cells are used then there is too much numerical dissipation introduced so keeping the number of x-cells 
%% at this order of magnitude is suggested. Changing the number of cells in the y-direction won't do much 
%% because this is a purely 1 dimensional flow so no v-velocity will be introduced.

clear all;
clc;
close all;

tic

%% Initialize U at t = 0s

% Left state primitive variables
rhol = 1.0;
Pl = 1.0;
ul = 0;
vl = 0;

% Right state primitive variables
rhor = 0.125;
Pr = 0.1;
ur = 0;
vr = 0;
gamma = 1.4;

% Convert primitive variables to conserved variables

UL = [rhol; rhol*ul; rhol*vl; Pl/(gamma-1) + 1/2*rhol*(ul^2 + vl^2)];
UR = [rhor; rhor*ur; rhor*vr; Pr/(gamma-1) + 1/2*rhor*(ur^2 + vr^2)];

% Grid geometry and creation

% Grid discretization
dy = 1/9;
dx = 2/3999;
x = -1:dx:1;
y = 0:dy:1;

% Cell centers
xcenter = x+dx;
ycenter = y+dy;

[X,Y] = meshgrid(xcenter,ycenter);
X = X';
Y = Y';

% Face areas (uniform for all cells)
Sx = dx;
Sy = dy;

% Face Normals (quadrilateral cells)
nx_R= 1;
nx_L = -1;
ny_T = 1;
ny_B = -1;

nx_T = 0;
nx_B = 0;
ny_L = 0;
ny_R = 0;


% Number of cells in each direction.
Nx = length(x);
Ny = length(y);

% Create empty matrices to store initial U and delta_U and new U that is computed at
% every timestep
U_initial = zeros(4,Nx,Ny);
U = zeros(4,Nx,Ny);
dU = zeros(4,Nx,Ny);

% Initialize U to be used at t = 0
for j = 1:Ny
    for i = 1:Nx
        if i <= Nx/2
            U_initial(:,i,j) = UL;
        else
            U_initial(:,i,j) = UR;
        end
    end
end

U = U_initial;

% Time limits and CFL
t = 0;
tfinal = 0.5;
CFL = 0.9;
n = 0;

% Start main time loop. 

while t <= tfinal

    dt = Calculate_dt(U, dx, dy, Nx, Ny, CFL);

    for j = 2:Ny-1
        for i = 2:Nx-1

            % Calculate fluxes

            FR = FP(U(:,i,j), nx_R, ny_R) + FM(U(:,i+1,j), nx_R, ny_R); 
            FT = FP(U(:,i,j), nx_T, ny_T) + FM(U(:,i,j+1), nx_T, ny_T);
            FL = FP(U(:,i,j), nx_L, ny_L) + FM(U(:,i-1,j), nx_L, ny_L);
            FB = FP(U(:,i,j), nx_B, ny_B) + FM(U(:,i,j-1), nx_B, ny_B);
        
            % Boundary Conditions for top and bottom of tube. Instead of
            % using ghost cells the pressure is enforced.            

            P = (gamma - 1)*(U(4,i,j) - 1/2*U(1,i,j)*((U(2,i,j)/U(1,i,j))^2 + (U(3,i,j)/U(1,i,j))^2));

            if j == 2 
                FB = [0; 0; -P; 0]; 
            end

            if j == Ny-1
                FT = [0; 0; P; 0];                
            end               
        
            % Update U
            dU(:,i,j) = -dt/(dx*dy)*(FR*Sy + FT*Sx + FL*Sy + FB*Sx);   

        end
    end
    
    
    % Update time and solution
    U = U + dU;
    t = t + dt;
    n = n+1;

    % Plot every 50th solve
    if rem(n, 50) == 0
        Postprocessing(U, Nx, Ny, X, Y, x);
    end
            
    % Reaffirm Boundary Conditions at x limits.

    U(:,1,:) = U_initial(:,1,:);
    U(:,Nx,:) = U_initial(:,Nx,:);
    
end

%% Visualization
Postprocessing(U, Nx, Ny, X, Y, x);

%% Function that calculates F+ flux
function [Fplus] = FP(Ui, nx, ny)

gamma = 1.4;

%Convert conserved variables to primitive variables.
V = consToPrim(Ui);

rho = V(1);
u = V(2);
v = V(3);
P = V(4);

% Speed of sound
a = sqrt(gamma*P/rho);

% Rotated velocity vector
uprime = u*nx + v*ny;

% Positive eigenvalues
l = [1/2*(uprime - a + abs(uprime - a)), 1/2*(uprime + abs(uprime)), 1/2*(uprime + abs(uprime)), 1/2*(uprime + a + abs(uprime + a))];

% M matrix (R Lambda R^(-1))
M = [l(2), rho*nx/(2*a)*(-l(1) + l(4)), rho*ny/(2*a)*(-l(1) + l(4)), 1/(2*a^2)*(l(1) - 2*l(2) + l(4));
    0, 1/2*(l(1)*nx^2 + 2*l(3)*ny^2 + l(4)*nx^2), nx*ny/2*(l(1) - 2*l(3) + l(4)), nx/(2*a*rho)*(-l(1) + l(4));
    0, nx*ny/2*(l(1) - 2*l(3) + l(4)), 1/2*(l(1)*ny^2 + 2*l(3)*nx^2 + l(4)*ny^2), ny/(2*a*rho)*(-l(1) + l(4));
    0, a*rho*nx/2*(-l(1) + l(4)), a*rho*ny/2*(-l(1) + l(4)), 1/2*(l(1) + l(4))];

% S Matrix
dvdu = [1, 0, 0, 0;
    -u/rho, 1/rho, 0, 0;
    -v/rho 0, 1/rho, 0;
    1/2*(gamma-1)*(u^2 + v^2), -(gamma-1)*u, -(gamma-1)*v, gamma - 1];

% S^(-1) matrix
dudv = [1, 0, 0, 0;
    u, rho, 0, 0;
    v, 0, rho, 0;
    1/2*(u^2+v^2), rho*u, rho*v, 1/(gamma-1)];

% Flux per unit length (A^(+)*U_(i))
Fplus = dudv*M*dvdu*Ui;

end

%% Function that calculates F- flux
function [Fminus] = FM(Uii, nx, ny)

gamma = 1.4;

%Convert conserved variables to primitive variables.
V = consToPrim(Uii);

rho = V(1);
u = V(2);
v = V(3);
P = V(4);

% Speed of sound
a = sqrt(gamma*P/rho);

% Rotated velocity vector
uprime = u*nx + v*ny;
l = [1/2*(uprime - a - abs(uprime - a)), 1/2*(uprime - abs(uprime)), 1/2*(uprime - abs(uprime)), 1/2*(uprime + a - abs(uprime + a))];

% M matrix (R Lambda R^(-1))
M = [l(2), rho*nx/(2*a)*(-l(1) + l(4)), rho*ny/(2*a)*(-l(1) + l(4)), 1/(2*a^2)*(l(1) - 2*l(2) + l(4));
    0, 1/2*(l(1)*nx^2 + 2*l(3)*ny^2 + l(4)*nx^2), nx*ny/2*(l(1) - 2*l(3) + l(4)), nx/(2*a*rho)*(-l(1) + l(4));
    0, nx*ny/2*(l(1) - 2*l(3) + l(4)), 1/2*(l(1)*ny^2 + 2*l(3)*nx^2 + l(4)*ny^2), ny/(2*a*rho)*(-l(1) + l(4));
    0, a*rho*nx/2*(-l(1) + l(4)), a*rho*ny/2*(-l(1) + l(4)), 1/2*(l(1) + l(4))];

% S Matrix
dvdu = [1, 0, 0, 0;
    -u/rho, 1/rho, 0, 0;
    -v/rho 0, 1/rho, 0;
    1/2*(gamma-1)*(u^2 + v^2), -(gamma-1)*u, -(gamma-1)*v, gamma - 1];

% S^(-1) matrix
dudv = [1, 0, 0, 0;
    u, rho, 0, 0;
    v, 0, rho, 0;
    1/2*(u^2+v^2), rho*u, rho*v, 1/(gamma-1)];

% Flux per unit length (A^(-)*U_(i+1))
Fminus = dudv*M*dvdu*Uii;

end

%% Function that converts conserved variables to primitive variables
function [W] = consToPrim(U)
           
    gamma = 1.4;    

    % Extract conserved variables from U
    rho = U(1);          % Density
    rhou = U(2);         % rho * u
    rhov = U(3);         % rho * v
    E = U(4);            % Total energy

    % Calculate primitive variables
    W = zeros(4,1);
    W(1) = rho;                 % Density is directly available
    W(2) = rhou / rho;          % Velocity in x-direction: u = (rho * u) / rho
    W(3) = rhov / rho;          % Velocity in y-direction: v = (rho * v) / rho

    % Total energy equation to calculate pressure
    kinetic_energy = 0.5 * rho * (W(2)^2 + W(3)^2);
    W(4) = (gamma - 1) * (E - kinetic_energy);  % Pressure: p = (gamma - 1) * (E - 0.5*rho*(u^2 + v^2))

end

%% Function that calculates dt using CFL condition

function [dt] = Calculate_dt(U, dx, dy, Nx, Ny, CFL)

    gamma = 1.4;
    int_dt = zeros(Nx,Ny);
    
    for i = 1:Nx
        for j = 1:Ny

            V = consToPrim(U(:,i,j));
            c = sqrt(gamma*V(4)/V(1));

            int_dt(i,j) =  CFL/( abs(V(2)/dx) + abs(V(3)/dy) + c*sqrt(1/dx^2 + 1/dy^2) );

        end
    end

    dt = min(min(int_dt));
    
end

%% Function that plots results
function [] = Postprocessing(U, Nx, Ny, X, Y, x)

V = zeros(4,Nx,Ny);

for i = 1:Nx
    for j = 1:Ny
        V(:,i,j) = consToPrim(U(:,i,j));
    end
end

rho = squeeze(V(1,:,:));
u = squeeze(V(2,:,:));
v = squeeze(V(3,:,:));
p = squeeze(V(4,:,:));

% 2D contour plot
figure(1);
tiledlayout(4,1);

nexttile
contourf(X,Y,rho);
xlabel('x');
ylabel('y');
grid on;
grid minor;
colormap jet;
colorbar;
title('Density')

nexttile
contourf(X,Y,u);
xlabel('x');
ylabel('y');
grid on;
grid minor;
colormap jet;
colorbar;
title('u-velocity')

nexttile
contourf(X,Y,v);
xlabel('x');
ylabel('y');
grid on;
grid minor;
colormap jet;
colorbar;
title('v-velocity')

nexttile
contourf(X,Y,p);
xlabel('x');
ylabel('y');
grid on;
grid minor;
colormap jet;
colorbar;
title('Pressure')

% 1D line plot
figure(2);
tiledlayout(4,1);

nexttile
a=plot(x,rho(:,5),'-r');
a.LineWidth = 1;
xlabel('x');
ylabel('\rho');
grid on;
grid minor;

nexttile
a=plot(x,u(:,5),'-r');
a.LineWidth = 1;
xlabel('x');
ylabel('u');
grid on;
grid minor;

nexttile
a=plot(x,v(:,5),'-r');
a.LineWidth = 1;
xlabel('x');
ylabel('v');
grid on;
grid minor;

nexttile
a=plot(x,p(:,5),'-r');
a.LineWidth = 1;
xlabel('x');
ylabel('P');
grid on;
grid minor;

end
