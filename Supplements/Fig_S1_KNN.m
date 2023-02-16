%Generalization error of KNN regression

close all;
clear all;


R = 10; % number of repeats
range_N = [3 10 30 100 300 1000 3000 10000]; % # of dimensions

mse_train = zeros(R,length(range_N));
mse_test = zeros(R,length(range_N));

N_y_t = 1; %output dimension
P_test = 1000;

SNR = inf; 

if SNR == inf
    variance_w = 1;
    variance_e = 0;
else
    variance_w = SNR/(SNR + 1);
    variance_e = 1/(SNR + 1);
end


K=1; % number of nearest neighbors

parfor repeat = 1:R
    
    mse_train_row = zeros(1,length(range_N)); 
    mse_test_row = zeros(1,length(range_N));
    
    for counter = 1:length(range_N)
        N_x_t = range_N(counter);
        P = N_x_t; % keeping alpha = 1
        train_error = zeros (1,P);
        test_error = zeros(1,P_test);

        w_t = normrnd(0,variance_w^0.5,[N_x_t,N_y_t]);

        %Generate patterns

        x_t_input = normrnd(0,(1/N_x_t)^0.5,[P,N_x_t]);
        noise = normrnd(0,variance_e^0.5,[P,N_y_t]);
        y_t_output = x_t_input*w_t + noise;


        x_t_input_new = normrnd(0,(1/N_x_t)^0.5,[P_test,N_x_t]);
        noise1 = normrnd(0,variance_e^0.5,[P_test,N_y_t]);
        y_t_output_new = x_t_input_new*w_t+ noise1;


        %Distance matrix

        D_train = pdist2(x_t_input,x_t_input);
        D_test = pdist2(x_t_input_new,x_t_input);

        %Train error
        for n = 1:P
            [B,I] = mink(D_train(n,:),K);
            retrieved_x = mean(x_t_input(I,:));
            retrieved_y = mean(y_t_output(I));
            train_error(n) = (y_t_output(n) - retrieved_y)^2;
        end

        %Test error
        for n = 1:P_test
            [B,I] = mink(D_test(n,:),K);
            retrieved_x = mean(x_t_input(I,:));
            retrieved_y = mean(y_t_output(I));
            test_error(n) = (y_t_output_new(n) - retrieved_y)^2;
        end

        mse_train_row(counter) = mean(train_error);
        mse_test_row(counter) = mean(test_error);

    end
        mse_train(repeat,:) = mse_train_row
        mse_test(repeat,:) = mse_test_row
end


figure(1)

plot(log10(range_N),mean(mse_test,1),'ro-','LineWidth',3)

xt = get(gca, 'XTick');
set(gca, 'FontSize', 20)
yt = get(gca, 'YTick');
set(gca, 'FontSize', 20)
ylim([0 2.5])