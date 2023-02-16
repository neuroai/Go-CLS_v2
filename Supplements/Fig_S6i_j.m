close all
clear all

nepoch = 2000;
learnrate = 0.005;
N_x_t = 100;
N_y_t = 1;
P=100;
M = 5000; %num of units in notebook

Et_big_early_stop = [];
Eg_big_early_stop = [];
Et_big = [];
Eg_big = [];

Fig3j_data = zeros(2,2);

Fig3k_data = zeros(2,2);


SNR_vec = [0.6 1000];


Notebook_Train = (P-1)/(M-1); % Analytial solution for notebook training error

for count = 1:size(SNR_vec,2)
    SNR = SNR_vec(count);
    %Analytical solution
    Eg = [];
    Et = [];
    lr = learnrate;
    
    %use normalized variances
    if SNR == inf
        variance_w = 1;
        variance_e = 0;
    else
        variance_w = SNR/(SNR + 1);
        variance_e = 1/(SNR + 1);
    end
    
    alpha = P/N_x_t; % number of examples divided by input dimension
    
    % Analytical curves for training and testing errors
    for t = 1:1:nepoch
        
        %correct integral
        train = @(lam) ( ( ( ((alpha^0.5+1).^2 - lam) .* (lam - (alpha^0.5-1).^2)  ).^0.5) ./  (lam*2*pi)  ).*  (  lam.*variance_w + variance_e  ).*exp(-2*lam.*t./(1./lr)) ;
        Et = [Et (1/alpha)*(integral(train,(alpha^0.5-1)^2,(alpha^0.5+1)^2) + (alpha<1)*(1 - alpha)* variance_e ) + (1-1/alpha)*variance_e];
        
        test = @(lam) ( ( ( ((alpha^0.5+1).^2 - lam) .* (lam - (alpha^0.5-1).^2)  ).^0.5) ./  (lam*2*pi) ).*(exp(-2*lam*t/(1/lr)) + ((1-exp(-lam*t/(1/lr))).^2)./(lam*SNR));
        Eg = [Eg variance_w*(integral(test,(alpha^0.5-1)^2,(alpha^0.5+1)^2) + (alpha<1)* (1 - alpha) + 1/SNR)];
        
    end
    
    % Early stopping curves
    [mm, pp] = min(Eg);
    Eg_early_stop = Eg;
    Eg_early_stop(pp+1:end) = Eg(pp);
    
    Et_early_stop = Et;
    Et_early_stop(pp+1:end) = Et(pp);
    
    % Pick better training error value and create memory generalization
    % scores
    better_train_no_early_stop = min(Et,ones(1,nepoch)*Notebook_Train);
    control_curve = (Et(1) -  better_train_no_early_stop)/Et(1);
    lesion_curve = (Et(1) - Et)/Et(1);
    
    
    better_train_yes_early_stop = min(Et_early_stop,ones(1,nepoch)*Notebook_Train);
    control_curve_early_stop = (Et_early_stop(1) -  better_train_yes_early_stop)/Et_early_stop(1);
    lesion_curve_early_stop = (Et_early_stop(1) - Et_early_stop)/Et_early_stop(1);
    
    
    control_Eg_curve = (Eg(1) -  Eg)/Eg(1);
    control_Eg_curve_early_stop = (Eg_early_stop(1) -  Eg_early_stop)/Eg_early_stop(1);
    
    Fig3j_data(count,1) = control_curve_early_stop(end);
    Fig3j_data(count,2) = control_Eg_curve_early_stop(end);

    Fig3k_data(count,1) = control_curve_early_stop(end);
    Fig3k_data(count,2) = lesion_curve_early_stop(end);
    
end


figure(1)

x = [[0.7,1.3];... 
     [2.7, 3.3]];

data = Fig3j_data;

     
f=bar(x,data*100);
f(1).BarWidth  = 3.2;
xlim([0 4])
ylim([0 120])
hold on

ax = gca;
ax.XTick = [1 3];
ax.YTick = [0 20 40 60 80 100];
xax = ax.XAxis;  
set(xax,'TickDirection','out')
set(gca,'box','off')
set(gcf,'position',[600,400,340,210])
set(gca, 'FontSize', 16)


ax.XTickLabel=[{'Discriminator'} {'Generalizer'}]; 


figure(2)

x = [[0.7,1.3];... 
     [2.7, 3.3]];

data = Fig3k_data;

     
f=bar(x,data*100);
f(1).BarWidth  = 3.2;
xlim([0 4])
ylim([0 120])
hold on

ax = gca;
ax.XTick = [1 3];
ax.YTick = [0 20 40 60 80 100];

xax = ax.XAxis;  
set(xax,'TickDirection','out')
set(gca,'box','off')
set(gcf,'position',[600,100,340,210])
set(gca, 'FontSize', 16)


ax.XTickLabel=[{'Discriminator'} {'Generalizer'}]; 
