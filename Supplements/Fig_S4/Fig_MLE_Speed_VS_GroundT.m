%% Clear workspace and initialize random number generator

clear all;
close all;
RandStream.setGlobalStream(RandStream('mt19937ar','seed',sum(100*clock)));

%% Construct an ensemble of teacher-generated student data.

num_teachers = 10;
N = 100;
P = 100;
std_X = sqrt(1/N);

SNRs = [0.05 4 100];

num_SNRs = length(SNRs);

wbars = NaN(N,num_SNRs,num_teachers);
Xs = NaN(P,N,num_SNRs,num_teachers);
etas = NaN(P,num_SNRs,num_teachers);
ys = NaN(P,num_SNRs,num_teachers);
for s = 1:num_SNRs;
    S = SNRs(s);
    std_noise = sqrt(1/(1+S));
    std_weights = sqrt(S/(1+S));
    for t = 1:num_teachers;
        wbars(:,s,t) = std_weights*randn(N,1);

        Xs(:,:,s,t) = std_X*randn(P,N);
        etas(:,s,t) = std_noise*randn(P,1);
        ys(:,s,t) = Xs(:,:,s,t)*wbars(:,s,t)+etas(:,s,t);
    end;
end;

var_ys = squeeze(var(ys,0,1));
var_ws = squeeze(var(wbars,0,1));
var_noises = squeeze(var(etas,0,1));
empirical_SNRs = (var_ys-var_noises)./var_noises;

mean_empirical_SNRs = mean(empirical_SNRs,2);
std_empirical_SNRs = std(empirical_SNRs,0,2);

%% Do an SVD decomposition on the data matrix.


Us = NaN(P,P,num_SNRs,num_teachers);
Ss = NaN(P,N,num_SNRs,num_teachers);
Vs = NaN(N,N,num_SNRs,num_teachers);
Lambdas = NaN(P,P,num_SNRs,num_teachers);
inv_Lambdas = NaN(P,P,num_SNRs,num_teachers);
for s = 1:num_SNRs;
    strcat('On SNR:',num2str(s))

    for t = 1:num_teachers;
        [Us(:,:,s,t) Ss(:,:,s,t) Vs(:,:,s,t)] = svd(Xs(:,:,s,t));
        Lambdas(:,:,s,t) = Ss(:,:,s,t)*Ss(:,:,s,t)';
        inv_Lambdas(:,:,s,t) = pinv(Lambdas(:,:,s,t));
    end;
end;

%% Evaluate the log-likelihood function for each dataset using the SVD decomposition
% More numerically efficient version.

tic;
num_components = min(N,P);


SNR_vector = 2.^(-6:0.02:7);
num_step_SNR = length(SNR_vector);

log_likelihood_function = NaN(num_step_SNR,num_SNRs,num_teachers);
max_log_likelihood = NaN(num_SNRs,num_teachers);
estimated_SNRs = NaN(num_SNRs,num_teachers);
for s = 1:num_SNRs;
    strcat('On SNR:',num2str(s))
    toc
    for t = 1:num_teachers;
        y_modes = Us(:,:,s,t)'*ys(:,s,t);
        for i = 1:num_step_SNR;
            inv_C_matrix_modes = (1+SNR_vector(i))*eye(P);
            A1 = (P-num_components)/2*log(1+SNR_vector(i));
            A2 = 0;
            for j = 1:num_components;
                inv_C_matrix_modes(j,j) = (1+SNR_vector(i))/(1+SNR_vector(i)*Lambdas(j,j,s,t));
                A1 = A1 + 1/2*log(inv_C_matrix_modes(j,j));
                A2 = A2 - 1/2*y_modes(j)^2*inv_C_matrix_modes(j,j);
            end;
            if gt(P,num_components)
                for j = 1:P;
                    A2 = A2 - 1/2*y_modes(j)^2*inv_C_matrix_modes(j,j);
                end;
            end;
            %inv_C_matrix_modes = (1+SNR_vector(i))*inv(eye(P)+SNR_vector(i)*Lambdas(:,:,s,t));
            %log_likelihood_function(i,s,t) = 1/2*log(det(inv_C_matrix_modes)) - 1/2*y_modes'*inv_C_matrix_modes*y_modes;
            log_likelihood_function(i,s,t) = A1 + A2;
            %log_likelihood_function(i,s,t) = -1/2*ys(:,s,t)'*inv_C_matrix*ys(:,s,t);
            %log_likelihood_function(i,s,t) = log(det(inv_C_matrix));
        end;
        [max_log_likelihood(s,t) tmp_idx] = max(log_likelihood_function(:,s,t));
        estimated_SNRs(s,t) = SNR_vector(tmp_idx);
    end;
end;


nepoch = 2000;
learnrate = 0.005;
N_x_t = 100;
N_y_t = 1;
P=100;

for count = 1:size(SNRs,2)
    disp(count)
    SNR = SNRs(count);
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

    % Analytical curves for training and testing errors, see supplementary
    % material for derivations

    for t = 1:1:nepoch

        train = @(lam) ( ( ( ((alpha^0.5+1).^2 - lam) .* (lam - (alpha^0.5-1).^2)  ).^0.5) ./  (lam*2*pi)  ).*  (  lam.*variance_w + variance_e  ).*exp(-2*lam.*t./(1./lr)) ;
        Et = [Et (1/alpha)*(integral(train,(alpha^0.5-1)^2,(alpha^0.5+1)^2) + (alpha<1)*(1 - alpha)* variance_e ) + (1-1/alpha)*variance_e];

        test = @(lam) ( ( ( ((alpha^0.5+1).^2 - lam) .* (lam - (alpha^0.5-1).^2)  ).^0.5) ./  (lam*2*pi) ).*(exp(-2*lam*t/(1/lr)) + ((1-exp(-lam*t/(1/lr))).^2)./(lam*SNR));
        Eg = [Eg variance_w*(integral(test,(alpha^0.5-1)^2,(alpha^0.5+1)^2) + (alpha<1)* (1 - alpha) + 1/SNR)];

    end
    [mm, pp] = min(Eg);
    Eg_early_stop = Eg;
    Eg_early_stop(pp+1:end) = Eg(pp);

    Et_early_stop = Et;
    Et_early_stop(pp+1:end) = Et(pp);

    ES_time = [];
    for ind = 1:length(estimated_SNRs(count,:))
        Eg_MLE = [];
        Et_MLE = [];

        SNR = estimated_SNRs(count,ind);

        if SNR == inf
            variance_w = 1;
            variance_e = 0;
        else
            variance_w = SNR/(SNR + 1);
            variance_e = 1/(SNR + 1);
        end


        for t = 1:1:nepoch

            train = @(lam) ( ( ( ((alpha^0.5+1).^2 - lam) .* (lam - (alpha^0.5-1).^2)  ).^0.5) ./  (lam*2*pi)  ).*  (  lam.*variance_w + variance_e  ).*exp(-2*lam.*t./(1./lr)) ;
            Et_MLE = [Et_MLE (1/alpha)*(integral(train,(alpha^0.5-1)^2,(alpha^0.5+1)^2) + (alpha<1)*(1 - alpha)* variance_e ) + (1-1/alpha)*variance_e];

            test = @(lam) ( ( ( ((alpha^0.5+1).^2 - lam) .* (lam - (alpha^0.5-1).^2)  ).^0.5) ./  (lam*2*pi) ).*(exp(-2*lam*t/(1/lr)) + ((1-exp(-lam*t/(1/lr))).^2)./(lam*SNR));
            Eg_MLE = [Eg_MLE variance_w*(integral(test,(alpha^0.5-1)^2,(alpha^0.5+1)^2) + (alpha<1)* (1 - alpha) + 1/SNR)];

        end
        [mm, pp] = min(Eg_MLE);
        ES_time = [ES_time pp];


    end
    % Early stopping curves

    figure(count)

    hold on;

    for i =1:length(ES_time)
        min_p = ES_time(i);
        Eg_early_stop_MLE = Eg;
        Eg_early_stop_MLE(min_p+1:end) = Eg(min_p);
        Et_early_stop_MLE = Et;
        Et_early_stop_MLE(min_p+1:end) = Et(min_p);

        plot(1:1:nepoch,Et_early_stop_MLE,'-','color',[0 1 1 0.5],'LineWidth',1)
        plot(1:1:nepoch,Eg_early_stop_MLE,'-','color',[1 0 1 0.5],'LineWidth',1)
    end
    plot(1:1:nepoch,Et_early_stop,'-','color',[0 0 1],'LineWidth',3)
    plot(1:1:nepoch,Eg_early_stop,'-','color',[1 0 0],'LineWidth',3)


    set(gca, 'FontSize',12)
    set(gca, 'FontSize', 12)
    xlabel('Epoch')
    ylabel('Error')
    set(gca,'linewidth',1.5)
    ylim([0 2])
    set(gcf,'position',[100,100,350,290])
    saveas(gcf,strcat('MLE SNR=',num2str(SNRs(count)),'.pdf'));


end

%loading estimated SNR values from the learning rate approach (obtained by running the associated code for main Fig.2m,n)
file = load('N_10000.mat'); 
field_names = fieldnames(file);
first_field_name = field_names{1};
Est_SNRs = file.(first_field_name);

for count = 1:size(SNRs,2)
    disp(count)
    SNR = SNRs(count);
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

    % Analytical curves for training and testing errors, see supplementary
    % material for derivations

    for t = 1:1:nepoch

        train = @(lam) ( ( ( ((alpha^0.5+1).^2 - lam) .* (lam - (alpha^0.5-1).^2)  ).^0.5) ./  (lam*2*pi)  ).*  (  lam.*variance_w + variance_e  ).*exp(-2*lam.*t./(1./lr)) ;
        Et = [Et (1/alpha)*(integral(train,(alpha^0.5-1)^2,(alpha^0.5+1)^2) + (alpha<1)*(1 - alpha)* variance_e ) + (1-1/alpha)*variance_e];

        test = @(lam) ( ( ( ((alpha^0.5+1).^2 - lam) .* (lam - (alpha^0.5-1).^2)  ).^0.5) ./  (lam*2*pi) ).*(exp(-2*lam*t/(1/lr)) + ((1-exp(-lam*t/(1/lr))).^2)./(lam*SNR));
        Eg = [Eg variance_w*(integral(test,(alpha^0.5-1)^2,(alpha^0.5+1)^2) + (alpha<1)* (1 - alpha) + 1/SNR)];

    end
    [mm, pp] = min(Eg);
    Eg_early_stop = Eg;
    Eg_early_stop(pp+1:end) = Eg(pp);

    Et_early_stop = Et;
    Et_early_stop(pp+1:end) = Et(pp);
    
    ES_time = [];
    for ind = 1:10
        Eg_LS = [];
        Et_LS = [];
        SNR_est = Est_SNRs(1+(count-1)*10:10+(count-1)*10);
        SNR = SNR_est(ind);

        if SNR == inf
            variance_w = 1;
            variance_e = 0;
        else
            variance_w = SNR/(SNR + 1);
            variance_e = 1/(SNR + 1);
        end


        for t = 1:1:nepoch

            train = @(lam) ( ( ( ((alpha^0.5+1).^2 - lam) .* (lam - (alpha^0.5-1).^2)  ).^0.5) ./  (lam*2*pi)  ).*  (  lam.*variance_w + variance_e  ).*exp(-2*lam.*t./(1./lr)) ;
            Et_LS = [Et_LS (1/alpha)*(integral(train,(alpha^0.5-1)^2,(alpha^0.5+1)^2) + (alpha<1)*(1 - alpha)* variance_e ) + (1-1/alpha)*variance_e];

            test = @(lam) ( ( ( ((alpha^0.5+1).^2 - lam) .* (lam - (alpha^0.5-1).^2)  ).^0.5) ./  (lam*2*pi) ).*(exp(-2*lam*t/(1/lr)) + ((1-exp(-lam*t/(1/lr))).^2)./(lam*SNR));
            Eg_LS = [Eg_LS variance_w*(integral(test,(alpha^0.5-1)^2,(alpha^0.5+1)^2) + (alpha<1)* (1 - alpha) + 1/SNR)];

        end
        [mm, pp] = min(Eg_LS);
        ES_time = [ES_time pp];


    end
    % Early stopping curves

    figure(count+3)

    hold on;

    for i =1:length(ES_time)
        min_p = ES_time(i);
        Eg_early_stop_MLE = Eg;
        Eg_early_stop_MLE(min_p+1:end) = Eg(min_p);

        Et_early_stop_MLE = Et;
        Et_early_stop_MLE(min_p+1:end) = Et(min_p);

        plot(1:1:nepoch,Et_early_stop_MLE,'-','color',[0 1 1 0.5],'LineWidth',1)
        plot(1:1:nepoch,Eg_early_stop_MLE,'-','color',[1 0 1 0.5],'LineWidth',1)
    end
    plot(1:1:nepoch,Et_early_stop,'-','color',[0 0 1],'LineWidth',3)
    plot(1:1:nepoch,Eg_early_stop,'-','color',[1 0 0],'LineWidth',3)


    set(gca, 'FontSize',12)
    set(gca, 'FontSize', 12)
    xlabel('Epoch')
    ylabel('Error')
    set(gca,'linewidth',1.5)
    ylim([0 2])
    set(gcf,'position',[100,100,350,290])
    saveas(gcf,strcat('Learning_speed SNR=',num2str(SNRs(count)),'_',first_field_name,'.pdf'));


end

%% save figures
% figure(1)
% set(gcf,'position',[100,100,350,290])
% saveas(gcf,strcat('Fig_2g','.pdf'));
% figure(2)
% set(gcf,'position',[100,100,350,290])
% saveas(gcf,strcat('Fig_2h','.pdf'));



