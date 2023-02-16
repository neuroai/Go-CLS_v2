% code for Student-Teacher-Notebook framework
% diversity of amnesia curves

close all
clear all

saveplot = false;
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

SNR_vec = [0.01 0.1 0.3 1 8 inf];

Notebook_Train = (P-1)/(M-1); % Analytial solution for notebook training error

% Simulation 1, varying SNR

for count = 1:size(SNR_vec,2)

    SNR = SNR_vec(count);
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

    % Early stopping curves
    [min_Eg, pos] = min(Eg);
    Eg_ES = Eg;
    Eg_ES(pos+1:end) = Eg(pos);

    Et_ES = Et;
    Et_ES(pos+1:end) = Et(pos);

    % Pick better training error and create memory and generalization
    % scores
    better_Et = min(Et,ones(1,nepoch)*Notebook_Train);
    control_memory_score = (Et(1) -  better_Et)/Et(1);
    lesion_memory_score = (Et(1) - Et)/Et(1);

    %with early stopping
    better_Et_ES = min(Et_ES,ones(1,nepoch)*Notebook_Train);
    control_memory_score_ES = (Et_ES(1) -  better_Et_ES)/Et_ES(1);
    lesion_memory_score_ES = (Et_ES(1) - Et_ES)/Et_ES(1);


    control_generalization_score = (Eg(1) -  Eg)/Eg(1);
    control_generalization_score_ES = (Eg_ES(1) -  Eg_ES)/Eg_ES(1);

    figure(1)

    hold on;
    if SNR == inf
        % picking an early and late epoch as "recent" and 'remote'
        plot([1 1800],[control_memory_score_ES(1) control_memory_score_ES(1800)],'o-','color',[0 0 0])
    end
    if SNR ~= inf
        plot([1 1800],[lesion_memory_score_ES(1) lesion_memory_score_ES(1800)],'o-','color',[0 0 0])
    end
    set(gca, 'FontSize', 12)
    set(gca, 'FontSize', 12)
    xlabel('Epoch', 'FontSize',12)
    ylabel('Memory Score', 'FontSize',12)
    xlim([-300 nepoch + 100])
    ylim([-0.1 1.15])
    set(gca,'linewidth',1.5)

end

% Simulation 2, varying prior consolidation

for start_epoch = [8 20 40 60 2000] % amount of prior learning epochs
    SNR = 50;
    P = 300;
    Eg = [];
    Et = [];
    lr = learnrate;

    if SNR == inf
        variance_w = 1;
        variance_e = 0;
    else
        variance_w = SNR/(SNR + 1);
        variance_e = 1/(SNR + 1);
    end

    alpha = P/N_x_t; % number of examples divided by input dimension


    % Analytical curves for training and testing errors
    for t = 1:1:(nepoch + start_epoch)

        train = @(lam) ( ( ( ((alpha^0.5+1).^2 - lam) .* (lam - (alpha^0.5-1).^2)  ).^0.5) ./  (lam*2*pi)  ).*  (  lam.*variance_w + variance_e  ).*exp(-2*lam.*t./(1./lr)) ;
        Et = [Et (1/alpha)*(integral(train,(alpha^0.5-1)^2,(alpha^0.5+1)^2) + (alpha<1)*(1 - alpha)* variance_e ) + (1-1/alpha)*variance_e];

        test = @(lam) ( ( ( ((alpha^0.5+1).^2 - lam) .* (lam - (alpha^0.5-1).^2)  ).^0.5) ./  (lam*2*pi) ).*(exp(-2*lam*t/(1/lr)) + ((1-exp(-lam*t/(1/lr))).^2)./(lam*SNR));
        Eg = [Eg variance_w*(integral(test,(alpha^0.5-1)^2,(alpha^0.5+1)^2) + (alpha<1)* (1 - alpha) + 1/SNR)];

    end

    % Early stopping curves
    [min_Eg, pos] = min(Eg);
    Eg_ES = Eg;
    Eg_ES(pos+1:end) = Eg(pos);

    Et_ES = Et;
    Et_ES(pos+1:end) = Et(pos);

    % Pick better training error value and create memory and generalization
    % scores
    better_Et = min(Et,ones(1,nepoch + start_epoch)*Notebook_Train);
    control_memory_score = (Et(1) -  better_Et)/Et(1);
    lesion_memory_score = (Et(1) - Et)/Et(1);


    better_Et_ES = min(Et_ES,ones(1,nepoch + start_epoch)*Notebook_Train);
    control_memory_score_ES = (Et_ES(1) -  better_Et_ES)/Et_ES(1);
    lesion_memory_score_ES = (Et_ES(1) - Et_ES)/Et_ES(1);


    control_generalization_score = (Eg(1) -  Eg)/Eg(1);
    control_generalization_score_ES = (Eg_ES(1) -  Eg_ES)/Eg_ES(1); % this assumes only using student for generalization
    lesion_Eg_curve_early_stop = (Eg_ES(1) -  Eg_ES)/Eg_ES(1);


    figure(1)

    hold on;

    %"Recent" memory performance with prior rule-consistent consolidation can
    %be captured by the generalization error. That is, given certain amount
    %of prior learning, how well can the network predict new examples?
    %"Remote" memory performance is the training error at convergence.

    plot([1 1800],[lesion_Eg_curve_early_stop(start_epoch) lesion_memory_score_ES(end)],'o--','color',[0 0 0])

    set(gca, 'FontSize', 12)
    set(gca, 'FontSize', 12)
    xlabel('Epoch', 'FontSize',12)
    ylabel('Memory Score', 'FontSize',12)
    xlim([-300 nepoch + 100])
    ylim([-0.1 1.15])
    set(gca,'linewidth',1.5)

end


f = gca;
f.XTick = [1 1800];
f.XTickLabel = [{'Recent'} {'Remote'}];
set(gcf,'position',[100,100,400,600])



