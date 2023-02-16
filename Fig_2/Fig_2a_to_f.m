% Main code for Student-Teacher-Notebook framework
% Weinan Sun 10-10-2021

tic
close all
clear all

r_n = 20; % number of repeats
nepoch = 2000;
learnrate = 0.015;
N_x_t = 100; % teacher input dimension
N_y_t = 1; % teacher output dimension
P=100; % number of training examples
P_test = 1000; % number of testing examples

% According to SNR, set variances for teacher's weights (variance_w) and output noise (variance_e) that sum to 1
SNR = inf;

if SNR == inf
    variance_w = 1;
    variance_e = 0;
else
    variance_w = SNR/(SNR + 1);
    variance_e = 1/(SNR + 1);
end

% Student and teacher share the same dimensions
N_x_s = N_x_t;
N_y_s = N_y_t;

% Notebook parameters
% see Buhmann, Divko, and Schulten, 1989 for details regarding gamma and U terms

M = 2000; % num of units in notebook
a = 0.05; % notebook sparseness
gamma = 0.6;    % inhibtion parameter
U = -0.15;    % threshold for unit activation
ncycle = 9;   %  number of recurrent cycles


% Matrices for storing train error, test error, reactivation error (driven by notebook)
% Without early stopping
train_error_all = zeros(r_n,nepoch); % student train error
test_error_all = zeros(r_n,nepoch); % student test error
N_train_error_all = zeros(r_n,nepoch); % notebook train error
N_test_error_all = zeros(r_n,nepoch); % notebook test error

% With early stopping
train_error_early_stop_all = zeros(r_n,nepoch);
test_error_early_stop_all = zeros(r_n,nepoch);

%Run simulation for r_n times
for r = 1:r_n
    disp(r)
    rng(r); %set random seed for reproducibility
    
    %Errors
    error_train_vector = zeros(nepoch,1);
    error_test_vector = zeros(nepoch,1);
    error_react_vector = zeros(nepoch,1);
    
    %% Teacher Network
    W_t = normrnd(0,variance_w^0.5,[N_x_t,N_y_t]); % set teacher's weights with variance_w
    noise_train = normrnd(0,variance_e^0.5,[P,N_y_t]); % set the variance for label noise
    % Training data
    x_t_input = normrnd(0,(1/N_x_t)^0.5,[P,N_x_t]); % inputs
    y_t_output = x_t_input*W_t + noise_train; % outputs
    
    % Testing data
    noise_test = normrnd(0,variance_e^0.5,[P_test,N_y_t]); 
    x_t_input_test = normrnd(0,(1/N_x_t)^0.5,[P_test,N_x_t]);
    y_t_output_test = x_t_input_test*W_t + noise_test;
    
    %% Notebook Network
    % Generate P random binary indices (0 or 1) with sparseness a
    N_patterns = zeros(P,M);
    for n=1:P
        N_patterns(n,randperm(M,M*a))=1;
    end
    
    %Hebbian learning for notebook recurrent weights
    W_N = (N_patterns - a)'*(N_patterns - a)/(M*a*(1-a));
    W_N = W_N - gamma/(a*M);% add global inhibiton term, see Buhmann, Divko, and Schulten, 1989
    W_N = W_N.*~eye(size(W_N)); % no self connection
    
    % Hebbian learning for Notebook-Student weights (bidirectional)

    % Notebook to student weights, for reactivating student
    W_N_S_Lin = (N_patterns-a)'*x_t_input/(M*a*(1-a));
    W_N_S_Lout = (N_patterns-a)'*y_t_output/(M*a*(1-a));
    % Student to notebook weights, for providing partial cues
    W_S_N_Lin = x_t_input'*(N_patterns-a);
    W_S_N_Lout = y_t_output'*(N_patterns-a);
    
    %% Student Network
    W_s = normrnd(0,0^0.5,[N_x_s,N_y_s]); % set students weights, with zero weight initialization
    
    %% Generate offline training data from notebook reactivations
    N_patterns_reactivated = zeros(P,M,nepoch,'logical'); % array for storing retrieved notebook patterns, pre-calculating all epochs for speed considerations
    
    parfor m = 1:nepoch
        %for m = 1:nepoch % regular for-loop 
        
        %% Notebook pattern completion through recurrent dynamis
        % Code below simulates hippocampal offline spontanenous
        % reactivations by seeding the initial notebook state with a random
        % binary pattern, then notebook goes through a two-step retrieval
        % process: (1) Retrieving a pattern using dynamic threshold to
        % ensure a pattern with sparseness a is retrieved (otherwise a silent
        % attractor will dominate retrieval). (2) Using the
        % retrieved pattern from (1) to seed a second round of pattern
        % completion using a fixed-threshold method (along with a global
        % inhibition term during encoding), so the retrieved patterns are
        % not forced to have a fixed sparseness, in addition, there is a
        % "silent  state" attractor when the seeding pattern lies far away
        % from any of the encoded patterns.
        
        % Start recurrent cycles with dynamic threshold
        Activity_dyn_t = zeros(P, M);
        
        % First round of pattern completion through recurrent activtion cycles given
        % random initial input.
        for cycle = 1:ncycle
            if cycle <=1
               clamp = 1;
            else
               clamp = 0;
            end
            rand_patt = (rand(P,M)<=a); % random seeding activity
            % Seeding notebook with random patterns
            M_input = Activity_dyn_t + (rand_patt*clamp);
            % Seeding notebook with original patterns
            % M_input = Activity_dyn_t + (N_patterns*clamp);
            M_current = M_input*W_N;
            % scale currents between 0 and 1
            scale = 1.0 ./ (max(M_current,[],2) - min(M_current,[],2));
            M_current = (M_current - min(M_current,[],2)) .* scale;
            % find threshold based on desired sparseness
            sorted_M_current = sort(M_current,2,'descend');
            t_ind = floor(size(Activity_dyn_t,2) * a);
            t_ind(t_ind<1) = 1;
            t = sorted_M_current(:,t_ind); % threshold for unit activations
            Activity_dyn_t = (M_current >=t);
        end
        
        % Second round of pattern completion, with fix threshold
        Activity_fix_t = zeros(P, M);
        for cycle = 1:ncycle
            if cycle <=1
                clamp = 1;
            else
                clamp = 0;
            end
            M_input = Activity_fix_t + Activity_dyn_t*clamp;
            M_current = M_input*W_N;
            Activity_fix_t = (M_current >= U); % U is the fixed threshold
        end
        N_patterns_reactivated(:,:,m)=Activity_fix_t;
    end
    
    % Seeding notebook with original notebook patterns for calculating
    % training error mediated by notebook (seeding notebook with student
    % input via Student's input to Notebook weights, once pattern completion
    % finishes, use the retrieved pattern to activate Student's output unit
    % via Notebook to Student's output weights.
    
    Activity_notebook_train = zeros(P, M);
    for cycle = 1:ncycle
        if cycle <=1
            clamp = 1;
        else
            clamp = 0;
        end
        seed_patt = x_t_input*W_S_N_Lin;
        M_input = Activity_notebook_train + (seed_patt*clamp);
        M_current = M_input*W_N;
        scale = 1.0 ./ (max(M_current,[],2) - min(M_current,[],2));
        M_current = (M_current - min(M_current,[],2)) .* scale;
        sorted_M_current = sort(M_current,2,'descend');
        t_ind = floor(size(Activity_notebook_train,2) * a);
        t_ind(t_ind<1) = 1;
        t = sorted_M_current(:,t_ind); 
        Activity_notebook_train = (M_current >=t);
    end
    N_S_output_train = Activity_notebook_train*W_N_S_Lout;
    % Notebook training error
    delta_N_train = y_t_output - N_S_output_train;
    error_N_train = sum(delta_N_train.^2)/P;
    % Since notebook errors stay constant throughout training,
    % populating each epoch with the same error value
    error_N_train_vector = ones(nepoch,1)*error_N_train; 
    N_train_error_all(r,:) = error_N_train_vector;
    
    % Notebook generalization error
    Activity_notebook_test = zeros(P_test, M);
    for cycle = 1:ncycle
        if cycle <=1
            clamp = 1;
        else
            clamp = 0;
        end
        seed_patt = x_t_input_test*W_S_N_Lin;
        M_input = Activity_notebook_test + (seed_patt*clamp);
        M_current = M_input*W_N;
        scale = 1.0 ./ (max(M_current,[],2) - min(M_current,[],2));
        M_current = (M_current - min(M_current,[],2)) .* scale;
        sorted_M_current = sort(M_current,2,'descend');
        t_ind = floor(size(Activity_notebook_test,2) * a);
        t_ind(t_ind<1) = 1;
        t = sorted_M_current(:,t_ind); 
        Activity_notebook_test = (M_current >=t);
    end
    N_S_output_test = Activity_notebook_test*W_N_S_Lout;
    % Notebook test error
    delta_N_test = y_t_output_test - N_S_output_test;
    error_N_test = sum(delta_N_test.^2)/P_test;
    % populating each epoch with the same error value
    error_N_test_vector = ones(nepoch,1)*error_N_test; 
    N_test_error_all(r,:) = error_N_test_vector;

    
%     N_patterns_reactivated_test = zeros(P_test,M,'logical');
    %% Student training through offline notebook reactivations at each epoch
    for m = 1:nepoch  
        
        N_S_input = N_patterns_reactivated(:,:,m)*W_N_S_Lin; % notebook reactivated student input activity
        N_S_output = N_patterns_reactivated(:,:,m)*W_N_S_Lout; % notebook reactivated student output activity
        N_S_prediction =  N_S_input*W_s; % student output prediction calculated by notebook reactivated input and student weights
        S_prediction =  x_t_input*W_s; % student output prediction calculated by true training inputs and student weights
        S_prediction_test =  x_t_input_test*W_s; % student output prediction calculated by true testing inputs and student weights
        
        % Train error
        delta_train = y_t_output - S_prediction;
        error_train = sum(delta_train.^2)/P;
        error_train_vector(m) = error_train;
        
        % Generalization error
        delta_test = y_t_output_test  - S_prediction_test;
        error_test = sum(delta_test.^2)/P_test;
        error_test_vector(m) = error_test;
        
        % Gradient descent
        w_delta = N_S_input'*N_S_output - N_S_input'*N_S_input*W_s;
        W_s = W_s + learnrate*w_delta;
    end
    
    train_error_all(r,:) = error_train_vector;
    test_error_all(r,:) = error_test_vector;
    
    % Early stopping 
    [min_v, min_p] = min(error_test_vector);
    train__error_early_stop = error_train_vector;
    train__error_early_stop (min_p+1:end) = error_train_vector (min_p);
    test_error_early_stop = error_test_vector;
    test_error_early_stop (min_p+1:end) = error_test_vector (min_p);
    train_error_early_stop_all(r,:) = train__error_early_stop;
    test_error_early_stop_all(r,:) = test_error_early_stop;
end

toc


%% Plotting
color_scheme = [137 152 193; 245 143 136]/255;
line_w = 2;
font_s = 12;

% Without early stopping
figure(1)
hold on
plot(1:nepoch,mean(train_error_all),'color',color_scheme(1,:),'LineWidth',line_w)
plot(1:nepoch,mean(test_error_all),'color',color_scheme(2,:),'LineWidth',line_w)
plot(1:nepoch,mean(N_train_error_all),'b--','LineWidth',2)
plot(1:nepoch,mean(N_test_error_all),'r--','LineWidth',2)

set(gca, 'FontSize', font_s)
set(gca, 'FontSize', font_s)
xlabel('Epoch','Color','k')
ylabel('Error','Color','k')
xlim([0 nepoch])
ylim([0 2])
set(gca,'linewidth',1)
set(gcf,'position',[100,100,350,300])

% Save plot
% saveas(gcf,strcat('Errors_No_ES_','SNR_',num2str(SNR),'_',date,'.pdf'));

% With early stopping
figure(2)
hold on
plot(1:nepoch,mean(train_error_early_stop_all),'color',color_scheme(1,:),'LineWidth',line_w)
plot(1:nepoch,mean(test_error_early_stop_all),'color',color_scheme(2,:),'LineWidth',line_w)
plot(1:nepoch,mean(N_train_error_all),'b--','LineWidth',line_w)
plot(1:nepoch,mean(N_test_error_all),'r--','LineWidth',line_w)

set(gca, 'FontSize', font_s)
set(gca, 'FontSize', font_s)
xlabel('Epoch','Color','k')
ylabel('Error','Color','k')
xlim([0 nepoch])
ylim([0 2])
set(gca,'linewidth',1)
set(gcf,'position',[600,100,350,300])

% Save plot
% saveas(gcf,strcat('Errors_ES_','SNR_',num2str(SNR),'_',date,'.pdf'));
