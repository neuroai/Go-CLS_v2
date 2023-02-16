%% Clear workspace and initialize random number generator

clear all;
RandStream.setGlobalStream(RandStream('mt19937ar','seed',sum(100*clock)));

%% Construct an ensemble of teacher-generated student data.

num_teachers = 1;
N = 1000;
P = 1000;
std_X = sqrt(1/N);

SNR_vec = -4:0.02:4;
SNRs = 2.^SNR_vec;

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

%% Plot basic stuff

%close all;

% figure(1)
% hold on;
% errorbar(SNRs,mean_empirical_SNRs,std_empirical_SNRs,'LineWidth',3)
% plot(SNRs, empirical_SNRs)
% hold off;
%set(gca,'XScale','log')
%set(gca,'YScale','log')

%% Evaluate the log-likelihood function for each dataset
% % DON'T USE THIS VERSION FOR LARGE N OR P.
% 
% tic;
% min_SNR = 0.02;
% max_SNR = 15;
% step_SNR = 0.02;
% SNR_vector = min_SNR:step_SNR:max_SNR;
% num_step_SNR = length(SNR_vector);
% 
% log_likelihood_function = NaN(num_step_SNR,num_SNRs,num_teachers);
% max_log_likelihood = NaN(num_SNRs,num_teachers);
% estimated_SNRs = NaN(num_SNRs,num_teachers);
% for s = 1:num_SNRs;
%     strcat('On SNR:',num2str(s))
%     toc
%     for t = 1:num_teachers;
%         for i = 1:num_step_SNR;
%             %inv_C_matrix = (1+SNR_vector(i))*(eye(P)-Xs(:,:,s,t)*inv(eye(N)/SNR_vector(i)+Xs(:,:,s,t)'*Xs(:,:,s,t))*Xs(:,:,s,t)');
%             inv_C_matrix = (1+SNR_vector(i))*inv(eye(P)+SNR_vector(i)*Xs(:,:,s,t)*Xs(:,:,s,t)');
%             log_likelihood_function(i,s,t) = 1/2*log(det(inv_C_matrix)) - 1/2*ys(:,s,t)'*inv_C_matrix*ys(:,s,t);
%             %log_likelihood_function(i,s,t) = -1/2*ys(:,s,t)'*inv_C_matrix*ys(:,s,t);
%             %log_likelihood_function(i,s,t) = log(det(inv_C_matrix));
%         end;
%         [max_log_likelihood(s,t) tmp_idx] = max(log_likelihood_function(:,s,t));
%         estimated_SNRs(s,t) = SNR_vector(tmp_idx);
%     end;
% end;


%% Plot the log_likelihood functions

%close all;
% 
% figure(2)
% for s = 1:num_SNRs;
%     for t = 1:2;
%         subplot(num_SNRs,2,(s-1)*2+t)
%         hold on;
%         plot(SNR_vector,log_likelihood_function(:,s,t));
%         plot(empirical_SNRs(s,t),max_log_likelihood(s,t),'ob')
%         plot(estimated_SNRs(s,t),max_log_likelihood(s,t),'or','LineWidth',2)
%     end;
% end;
% 
% figure(10)
% plot(empirical_SNRs(:),estimated_SNRs(:),'ok')

%% Do an SVD decomposition on the data matrix.


Us = NaN(P,P,num_SNRs,num_teachers);
Ss = NaN(P,N,num_SNRs,num_teachers);
Vs = NaN(N,N,num_SNRs,num_teachers);
Lambdas = NaN(P,P,num_SNRs,num_teachers);
inv_Lambdas = NaN(P,P,num_SNRs,num_teachers);
parfor s = 1:num_SNRs;
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


SNR_vector = 2.^(-6:0.02:6);
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

%% Plot the log_likelihood functions

%close all;

% figure(3)
% for s = 1:num_SNRs;
%     for t = 1:2;
%         subplot(num_SNRs,2,(s-1)*2+t)
%         hold on;
%         
%         plot(SNR_vector,log_likelihood_function(:,s,t));
%         
%         plot(SNR_vector(s),max_log_likelihood(s,t),'ob')
%         plot(estimated_SNRs(s,t),max_log_likelihood(s,t),'or','LineWidth',2)
%     end;
% end;

figure(400)
plot(log2(SNRs),log2(estimated_SNRs(:,1)),'ok')


set(gcf,'position',[500 500 300 300])
xlim([-6,6])
ylim([-6,6])

figure(500)
hold on
plot(SNR_vector,log_likelihood_function(:,end))
plot(SNRs(end),max_log_likelihood(end),'ob')
plot(estimated_SNRs(end),max_log_likelihood(end),'or','LineWidth',2)









