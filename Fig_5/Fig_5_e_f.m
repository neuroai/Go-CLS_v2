%Fig.5 e and f, noisy, complex, and partial observable teachers
tic
close all
clear all

r_n = 1000; % number of repeats
nepoch = 500;
learnrate = 0.1;
N_x_t = 100;
N_y_t = 1;
P=100;
p_test = 1000;

%Set student dimensions
N_x_s = N_x_t;
N_y_s = N_y_t;

% N_x_t_wide = 117 for a partial observability level matches the sin complex teacher (SNR = 5.74)
% For Fig.5e, set N_x_t_wide = 100 for Fig. 5f. See supplementary material,
% complex teacher section.
N_x_t_wide = N_x_t + 17;


Error_vector_big_noisy = zeros(r_n,nepoch);
gError_vector_big_noisy = zeros(r_n,nepoch);

Error_vector_big_com = zeros(r_n,nepoch);
gError_vector_big_com = zeros(r_n,nepoch);

Error_vector_big_partial = zeros(r_n,nepoch);
gError_vector_big_partial = zeros(r_n,nepoch);




parfor r = 1:r_n

    rng(r)

    Error_vector_noisy = zeros(nepoch,1);
    gError_vector_noisy = zeros(nepoch,1);


    Error_vector_com = zeros(nepoch,1);
    gError_vector_com = zeros(nepoch,1);

    Error_vector_partial = zeros(nepoch,1);
    gError_vector_partial = zeros(nepoch,1);



    %% Teacher Network

    w_t = normrnd(0,1^0.5,[N_x_t,N_y_t]);
    w_t_wide = normrnd(0,1^0.5,[N_x_t_wide,N_y_t]);

    %training data
    x_t_input = normrnd(0,(1/N_x_t)^0.5,[P,N_x_t]); %input patterns for noisy and complex teachers
    x_t_input_wide = normrnd(0,(1/N_x_t)^0.5,[P,N_x_t_wide]); %input patterns for partially observable teacher
    y_t_output_wide = x_t_input_wide*w_t_wide; %output of the partially observable teacher
    observable_mask = randperm(N_x_t_wide,100); % define the region of teacher input observable to the student
    x_t_input_observable = x_t_input_wide(:,observable_mask);
    y_t_output_complex = sin(x_t_input*w_t); % complex teacher output, using sine function for Fig. 5e, remove sine function for Fig. 5f.

    % Below are terms for calculating equivalent SNR of a complex teacher.
    % See supplementary material, complex teacher section for details.
    x_t_input_big = normrnd(0,(1/N_x_t)^0.5,[P*1000,N_x_t]);
    y_t_output_complex_big = sin(x_t_input_big*w_t);
    % Variance of the optimal linear weight for fitting the training data by a complex
    % teacher.
    variance_w_opt = var(inv(x_t_input_big'*x_t_input_big)*(x_t_input_big'*y_t_output_complex_big));
    % variance of the residue after linear fitting
    variance_c =  mean((y_t_output_complex - x_t_input*inv(x_t_input_big'*x_t_input_big)*(x_t_input_big'*y_t_output_complex_big)).^2);


    w_t_opt = normrnd(0,variance_w_opt^0.5,[N_x_t,N_y_t]); % set weight variance for the noisy teacher
    noise = normrnd(0,variance_c^0.5,[P,N_y_t]); % set training data noise variance for the noisy teacher
    noise1 = normrnd(0,variance_c^0.5,[p_test,N_y_t]);% set testing data noise variance for the noisy teacher
    y_t_output =x_t_input*w_t_opt + noise;  %output of noisy teacher

    %test sets
    x_t_input_new = normrnd(0,(1/N_x_t)^0.5,[p_test,N_x_t]);
    y_t_output_new = x_t_input_new*w_t_opt + noise1;
    y_t_output_new_complex = sin(x_t_input_new*w_t);

    x_t_input_wide_new = normrnd(0,(1/N_x_t)^0.5,[p_test,N_x_t_wide]);
    x_t_input_wide_new_observable = x_t_input_wide_new(:,observable_mask);
    y_t_output_wide_new = x_t_input_wide_new*w_t_wide;


    %% Student Network

    w_s_noisy = normrnd(0,0^0.5,[N_x_s,N_y_s]);
    w_s_complex = normrnd(0,0^0.5,[N_x_s,N_y_s]);
    w_s_partial = normrnd(0,0^0.5,[N_x_s,N_y_s]);




    for m = 1:nepoch



        y_s_output_noisy =  x_t_input*w_s_noisy;
        y_s_output_new_noisy =  x_t_input_new*w_s_noisy;

        y_s_output_batch_com =  x_t_input*w_s_complex;
        y_s_output_new_batch_com =  x_t_input_new*w_s_complex;

        y_s_output_batch_partial =  x_t_input_observable*w_s_partial;
        y_s_output_new_batch_partial =  x_t_input_wide_new_observable*w_s_partial;


        %noisy teacher train error
        Error_noisy = y_t_output - y_s_output_noisy;
        MSE_noisy = sum(Error_noisy.^2)/P;
        Error_vector_noisy(m) = MSE_noisy;

        %noisy teacher generalization error
        gError_noisy = y_t_output_new  - y_s_output_new_noisy;
        gMSE_noisy = sum(gError_noisy.^2)/p_test;
        gError_vector_noisy(m) = gMSE_noisy;


        %complex teacher train error 
        Error_com = y_t_output_complex - y_s_output_batch_com;
        MSE_com = sum(Error_com.^2)/P;
        Error_vector_com(m) = MSE_com;

        %complex teacher generalization error 
        gError_com = y_t_output_new_complex  - y_s_output_new_batch_com;
        gMSE_com = sum(gError_com.^2)/p_test;
        gError_vector_com(m) = gMSE_com;

        %partial observable teacher train error 
        Error_partial = y_t_output_wide - y_s_output_batch_partial;
        Cost_partial = sum(Error_partial.^2)/P;
        Error_vector_partial(m) = Cost_partial;

        %partial observable teacher generalization error 
        gError_partial = y_t_output_wide_new  - y_s_output_new_batch_partial;
        gCost_partial = sum(gError_partial.^2)/p_test;
        gError_vector_partial(m) = gCost_partial;

        % Weight updates through gradient descent for teach teacher
        w_delta_noisy = (x_t_input'*y_t_output - x_t_input'*x_t_input*w_s_noisy);
        w_s_noisy = w_s_noisy + learnrate*w_delta_noisy;

        w_delta_com = (x_t_input'*y_t_output_complex - x_t_input'*x_t_input*w_s_complex);
        w_s_complex = w_s_complex + learnrate*w_delta_com;

        w_delta_partial = (x_t_input_observable'*y_t_output_wide -  x_t_input_observable'* x_t_input_observable*w_s_partial);
        w_s_partial = w_s_partial + learnrate*w_delta_partial;
    end

    Error_vector_big_noisy(r,:) = Error_vector_noisy;
    gError_vector_big_noisy(r,:) = gError_vector_noisy;

    Error_vector_big_com(r,:) = Error_vector_com;
    gError_vector_big_com(r,:) = gError_vector_com;

    Error_vector_big_partial(r,:) = Error_vector_partial;
    gError_vector_big_partial(r,:) = gError_vector_partial;

end

toc

color_scheme = [137 152 193; 245 143 136]/255;
line_w = 1;
font_s = 12;

figure(1)
hold on


plot(1:nepoch,mean(Error_vector_big_noisy)/mean(Error_vector_big_noisy(:,1)),'r-')
plot(1:nepoch,mean(gError_vector_big_noisy)/mean(gError_vector_big_noisy(:,1)),'r-')


plot(1:nepoch,mean(Error_vector_big_com)/mean(Error_vector_big_com(:,1)),'k-')
plot(1:nepoch,mean(gError_vector_big_com)/mean(gError_vector_big_com(:,1)),'k-')

plot(1:nepoch,mean(Error_vector_big_partial)/mean(Error_vector_big_partial(:,1)),'g-')
plot(1:nepoch,mean(gError_vector_big_partial)/mean(gError_vector_big_partial(:,1)),'g-')



xt = get(gca, 'XTick');
set(gca, 'FontSize', font_s)
yt = get(gca, 'YTick');
set(gca, 'FontSize', font_s)
xlabel('Epoch','Color','k')
ylabel('Error','Color','k')
set(gcf,'position',[100,100,360,225])
xlim([0 nepoch])

% print(gcf,'complex_teacher_low_SNR.png','-dpng','-r600');








